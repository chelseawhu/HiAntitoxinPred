#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hiantitoxinpred.training import CNNRescueTrainingConfig, build_cnn_rescue_model  # noqa: E402
from hiantitoxinpred.utils import normalize_protein_sequence, resolve_device  # noqa: E402


def read_fasta(path: str | Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_id: str | None = None
    current_chunks: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, normalize_protein_sequence("".join(current_chunks))))
                current_id = line[1:].split()[0]
                current_chunks = []
            else:
                current_chunks.append(line)
    if current_id is not None:
        records.append((current_id, normalize_protein_sequence("".join(current_chunks))))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def align_hidden_to_sequence(hidden: np.ndarray, expected_length: int) -> np.ndarray:
    if hidden.shape[0] == expected_length:
        return hidden.astype(np.float32, copy=False)
    if hidden.shape[0] > expected_length:
        return hidden[:expected_length].astype(np.float32, copy=False)
    pad = np.zeros((expected_length - hidden.shape[0], hidden.shape[1]), dtype=np.float32)
    return np.concatenate([hidden.astype(np.float32, copy=False), pad], axis=0)


def extract_esm2_features(
    records: List[Tuple[str, str]],
    *,
    model_ref: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModel.from_pretrained(model_ref)
    model = model.to(resolve_device(device))
    model.eval()
    special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids)), dtype=torch.long)

    output: List[Dict[str, Any]] = []
    ids = [item[0] for item in records]
    sequences = [item[1] for item in records]
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch_sequences = sequences[start:end]
            encoded = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"].bool()
            special_mask = torch.isin(input_ids, special_ids.to(model.device))
            valid_mask = attention_mask & (~special_mask)
            for offset in range(hidden.shape[0]):
                residue_mask = valid_mask[offset]
                if not residue_mask.any():
                    residue_mask = attention_mask[offset]
                per_residue = hidden[offset][residue_mask].detach().cpu().numpy().astype(np.float32)
                expected_length = len(batch_sequences[offset])
                per_residue = align_hidden_to_sequence(per_residue, expected_length)
                output.append(
                    {
                        "id": ids[start + offset],
                        "sequence": batch_sequences[offset],
                        "features": per_residue,
                        "length": int(per_residue.shape[0]),
                        "feature_dim": int(per_residue.shape[1]),
                    }
                )
    return output


def pad_feature_batch(feature_records: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(item["features"].shape[0]) for item in feature_records)
    feature_dim = int(feature_records[0]["features"].shape[1])
    batch = torch.zeros((len(feature_records), max_len, feature_dim), dtype=torch.float32)
    lengths = torch.zeros((len(feature_records),), dtype=torch.long)
    for idx, item in enumerate(feature_records):
        features = torch.as_tensor(item["features"], dtype=torch.float32)
        length = int(features.shape[0])
        batch[idx, :length, :] = features
        lengths[idx] = length
    return batch, lengths


def build_model(config_path: Path, feature_dim: int, checkpoint_path: Path, device: str) -> torch.nn.Module:
    config = load_json(config_path)
    overrides = dict(config["main_model_overrides"])
    overrides.update(
        {
            "esm_dim": int(feature_dim),
            "random_seed": 42,
            "deterministic": False,
            "use_onehot": False,
            "use_aaindex": False,
            "use_cnn": True,
            "use_transformer": True,
        }
    )
    training_config = CNNRescueTrainingConfig(**overrides)
    model = build_cnn_rescue_model(training_config, device)
    try:
        state = torch.load(checkpoint_path, map_location=resolve_device(device), weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=resolve_device(device))
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_threshold(cli_threshold: float | None, checkpoint_path: str | Path) -> Tuple[float, str]:
    """Pick the decision threshold: explicit CLI value > checkpoint provenance > 0.5.

    The released checkpoint is calibrated at a non-0.5 threshold (the validation-MaxF1
    value stored in its *_provenance.json). Defaulting to 0.5 would systematically
    under-call positives, so the recommended threshold travels with the checkpoint.
    """
    if cli_threshold is not None:
        return float(cli_threshold), "command-line --threshold"
    checkpoint_path = Path(checkpoint_path)
    provenance = checkpoint_path.with_name(checkpoint_path.stem + "_provenance.json")
    if provenance.exists():
        try:
            recommended = json.loads(provenance.read_text()).get("recommended_threshold_prob")
        except (ValueError, OSError):
            recommended = None
        if recommended is not None:
            return float(recommended), f"checkpoint provenance ({provenance.name})"
    return 0.5, "fallback default 0.5 (no checkpoint-specific threshold found)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict antitoxins from protein FASTA using HiAntitoxinPred.")
    parser.add_argument("--fasta", required=True, help="Input protein FASTA file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained HiAntitoxinPred checkpoint.")
    parser.add_argument(
        "--config",
        default="configs/model_runs/main_full_model_5seed.json",
        help="Model config used to build the network architecture.",
    )
    parser.add_argument(
        "--esm2-model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace model name or local path for ESM2-650M.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for the positive (antitoxin) call. If omitted, "
        "use the checkpoint's recommended_threshold_prob from its *_provenance.json, "
        "else fall back to 0.5 with a warning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold, threshold_source = resolve_threshold(args.threshold, args.checkpoint)
    if threshold_source.startswith("fallback"):
        print(
            "WARNING: no --threshold given and no recommended threshold found for this "
            "checkpoint; using 0.5, which may be wrong for this model. Calibrate the "
            "threshold on a labelled validation set.",
            file=sys.stderr,
        )
    print(f"Decision threshold: {threshold:.4f} (source: {threshold_source})")
    records = read_fasta(args.fasta)
    feature_records = extract_esm2_features(
        records,
        model_ref=str(args.esm2_model),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    feature_dim = int(feature_records[0]["feature_dim"])
    model = build_model(Path(args.config), feature_dim, Path(args.checkpoint), str(args.device))
    esm_batch, lengths = pad_feature_batch(feature_records)
    device = resolve_device(args.device)
    esm_batch = esm_batch.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        logits = model(esm_batch, None, None, esm_len=lengths).reshape(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    rows = []
    for item, prob, logit in zip(feature_records, probs, logits.detach().cpu().numpy()):
        rows.append(
            {
                "id": item["id"],
                "sequence": item["sequence"],
                "length": item["length"],
                "probability": float(prob),
                "logit": float(logit),
                "prediction": int(float(prob) >= threshold),
                "threshold": float(threshold),
                "threshold_source": threshold_source,
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
