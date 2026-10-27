"""Frozen local ProtT5 encoding with explicit precision and cache identity."""

import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
import torch
from .checkpoints import file_sha256
from .data import Protein, ALPHABET

FILES = (
    "config.json",
    "pytorch_model.bin",
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

REGISTERED_CHECKSUMS = {
    "config.json": "c93cc8c31433fb062cad1825a9c16cde962a2015c8a3b10cfdc5f76a85128db3",
    "pytorch_model.bin": "7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157",
    "special_tokens_map.json": "a4b8dfd3d9a75b0f6f9ddbcb52b147c7377f08aa9310e182829cda2a55684334",
    "spiece.model": "74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324",
    "tokenizer_config.json": "d3adf2fb21d1242b5c0b72602f02ce5b0c43f767c0f957e3b57a67b2f93bf4f2",
}


def cache_key(sequence, identity):
    return hashlib.sha256(
        json.dumps(
            dict(sequence=sequence, encoder=identity),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def residue_arrays(hidden, attention, special, lengths):
    h = np.asarray(hidden, dtype=np.float32)
    a = np.asarray(attention, dtype=bool)
    s = np.asarray(special, dtype=bool)
    if (
        h.ndim != 3
        or h.shape[2] != 1024
        or a.shape != h.shape[:2]
        or s.shape != a.shape
    ):
        raise ValueError("Invalid encoder output shape")
    mask = a & ~s
    if mask.sum(1).tolist() != list(lengths) or not np.isfinite(h).all():
        raise ValueError("Token lengths or finite encoding check failed")
    return [np.array(h[i][mask[i]], copy=True) for i in range(len(lengths))]


class ProtT5Encoder:
    def __init__(self, model_dir, *, device="cpu", cache_dir=None):
        from transformers import (
            T5EncoderModel,
            T5Tokenizer,
            __version__ as transformers_version,
        )

        self.device = torch.device(device)
        self.root = Path(model_dir)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: cpu or cuda")
        extra_tokenizers = [
            p.name
            for p in self.root.glob("tokenizer*.json")
            if p.name != "tokenizer_config.json"
        ]
        if (self.root / "added_tokens.json").exists():
            extra_tokenizers.append("added_tokens.json")
        if extra_tokenizers:
            raise ValueError(
                "Unregistered tokenizer files: " + ", ".join(sorted(extra_tokenizers))
            )
        hashes = {n: file_sha256(self.root / n) for n in FILES}
        if hashes != REGISTERED_CHECKSUMS:
            raise ValueError(
                "Encoder files differ from registered ProtT5; use the pinned encoder download"
            )
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.tokenizer = T5Tokenizer.from_pretrained(
            str(self.root), do_lower_case=False, local_files_only=True
        )
        self.model, info = T5EncoderModel.from_pretrained(
            str(self.root),
            torch_dtype=dtype,
            local_files_only=True,
            output_loading_info=True,
            use_safetensors=False,
        )
        if any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError("Encoder loading mismatch")
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        layout = {n: str(p.dtype) for n, p in self.model.named_parameters()}
        if self.model.config.d_model != 1024:
            raise ValueError("Expected ProtT5 width1024")
        if self.device.type == "cuda":
            fp32 = {n for n, d in layout.items() if d == "torch.float32"}
            if (
                len(fp32) != 24
                or not all(n.endswith(".wo.weight") for n in fp32)
                or sum(d == "torch.float16" for d in layout.values()) != 171
            ):
                raise ValueError(
                    "Expected original mixed171 FP16/24 FP32 wo layout; do not call model.half()"
                )
        elif any(d != "torch.float32" for d in layout.values()):
            raise ValueError("CPU encoder must use FP32")
        self.identity = dict(
            name="ProtT5-XL-U50-half-encoder",
            files=hashes,
            precision="cpu_fp32" if self.device.type == "cpu" else "cuda_fp16_fp32wo",
            parameter_dtypes=layout,
            torch=str(torch.__version__),
            transformers=transformers_version,
            preprocessing="cap200-before-UZOB-to-X; spaced tokens; exclude special/padding; float32 storage",
            version=1,
        )
        self.cache = None if cache_dir is None else Path(cache_dir)
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)

    def encode(self, sequences, *, batch_size=2):
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("Invalid encoder batch size")
        if not sequences:
            raise ValueError("Empty sequence list")
        for seq in sequences:
            if (
                not isinstance(seq, str)
                or not 1 <= len(seq) <= 200
                or set(seq) - ALPHABET
            ):
                raise ValueError("Pass validated, cap200 uppercase input sequences")
        result = [None] * len(sequences)
        with torch.inference_mode():
            for offset in range(0, len(sequences), batch_size):
                group = list(range(offset, min(offset + batch_size, len(sequences))))
                # Cache entries bind the original encoder batch, not just a
                # sequence. Partial hits must not regroup the remaining inputs.
                context = dict(
                    encoder=self.identity, batch_sequences=[sequences[i] for i in group]
                )
                keys = {i: cache_key(sequences[i], context) for i in group}
                missing = []
                for i in group:
                    key = keys[i]
                    if (
                        self.cache is not None
                        and (self.cache / (key + ".json")).exists()
                    ):
                        meta = json.loads((self.cache / (key + ".json")).read_text())
                        p = self.cache / (key + ".npy")
                        if meta.get("key") != key or file_sha256(p) != meta.get(
                            "sha256"
                        ):
                            raise ValueError("Cache identity/hash mismatch")
                        v = np.load(p, allow_pickle=False)
                        if (
                            v.shape != (len(sequences[i]), 1024)
                            or v.dtype != np.float32
                            or not np.isfinite(v).all()
                        ):
                            raise ValueError("Invalid cached features")
                        result[i] = v
                    else:
                        missing.append(i)
                if not missing:
                    continue
                strings = [
                    " ".join(Protein("", sequences[i], sequences[i]).encoder_sequence)
                    for i in group
                ]
                tokens = self.tokenizer(
                    strings,
                    padding=True,
                    truncation=False,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                special = tokens.pop("special_tokens_mask").numpy()
                attention = tokens["attention_mask"].numpy()
                hidden = self.model(
                    **{k: v.to(self.device) for k, v in tokens.items()}
                ).last_hidden_state
                values = residue_arrays(
                    hidden.float().cpu().numpy(),
                    attention,
                    special,
                    [len(sequences[i]) for i in group],
                )
                for i, v in zip(group, values):
                    if i not in missing:
                        continue
                    result[i] = v
                    if self.cache is not None:
                        key = keys[i]
                        p = self.cache / (key + ".npy")
                        with tempfile.NamedTemporaryFile(
                            dir=self.cache, delete=False
                        ) as f:
                            np.save(f, v, allow_pickle=False)
                            tmp = Path(f.name)
                        os.replace(tmp, p)
                        meta = dict(key=key, sha256=file_sha256(p), length=len(v))
                        with tempfile.NamedTemporaryFile(
                            mode="w", dir=self.cache, delete=False
                        ) as f:
                            json.dump(meta, f)
                            tmp = Path(f.name)
                        os.replace(tmp, self.cache / (key + ".json"))
        return result
