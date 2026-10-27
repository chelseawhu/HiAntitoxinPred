"""Small public entry points; training and held-out evaluation are separate."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from .artifacts import fetch_asset, extract_zip, load_registered_model
from .checkpoints import file_sha256
from .data import read_fasta, read_registered_main
from .encoder import ProtT5Encoder
from .features import FeatureStore, ResidueDataset
from .inference import predict_features
from .maia import explain
from .metrics import binary_metrics
from .model import HiAntitoxinPred, ModelConfig
from .training import TrainingConfig, fit, predict_dataset, save_predictions, write_json


def read_config(path):
    value = json.loads(Path(path).read_text())
    if set(value) != {"training", "model"}:
        raise ValueError("Configuration requires only training and model fields")
    return TrainingConfig(**value["training"]), ModelConfig(**value["model"])


def load_run_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    lock_file = checkpoint.parent / "SELECTION_LOCK.json"
    if checkpoint.name != "L.pt" or not lock_file.is_file():
        raise ValueError("A trained L.pt and its selection lock are required")
    lock = json.loads(lock_file.read_text())
    if file_sha256(checkpoint) != lock["weights_sha256"]:
        raise ValueError("Checkpoint differs from the selection lock")
    for name in ("COMPLETE.json", "RELOAD_VERIFICATION.json"):
        value = json.loads((checkpoint.parent / name).read_text())
        if not value["status"].startswith("PASS"):
            raise ValueError("Training/reload is not complete")
    model_config = json.loads((checkpoint.parent / "model_config.json").read_text())
    run_config = json.loads((checkpoint.parent / "CONFIG.json").read_text())
    if model_config != run_config["model"]:
        raise ValueError("Model configuration differs from the training record")
    model = HiAntitoxinPred(ModelConfig(**model_config)).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    return model.eval(), dict(lock, seed=run_config["seed"])


def _new_file(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _table(path, fields, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _runtime(args):
    model, identity = load_registered_model(
        args.models, seed=args.seed, device=args.device
    )
    encoder = ProtT5Encoder(args.encoder, device=args.device, cache_dir=args.cache)
    return model, identity, encoder


def predict(args):
    output = _new_file(args.output)
    metadata = _new_file(str(output) + ".json")
    proteins = read_fasta(args.fasta)
    model, identity, encoder = _runtime(args)
    features = encoder.encode(
        [p.sequence for p in proteins], batch_size=args.encoder_batch_size
    )
    logits = predict_features(
        model, [torch.from_numpy(x) for x in features], batch_size=args.batch_size
    )
    probability = torch.as_tensor(logits).sigmoid().numpy()
    rows = [
        dict(
            id=p.id,
            original_length=p.original_length,
            input_length=p.input_length,
            truncated=p.truncated,
            raw_logit=float(z),
            probability=float(prob),
            prediction=int(z >= 0),
        )
        for p, z, prob in zip(proteins, logits, probability)
    ]
    _table(output, list(rows[0]), rows)
    write_json(
        metadata,
        dict(
            model=identity,
            encoder=encoder.identity,
            padding_width=200,
            threshold=0.5,
            encoder_batch_size=args.encoder_batch_size,
            classifier_batch_size=args.batch_size,
            inputs=[
                dict(id=p.id, sequence=p.sequence, sequence_sha256=p.sequence_sha256)
                for p in proteins
            ],
        ),
    )


def maia(args):
    root = Path(args.output)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    proteins = read_fasta(args.fasta)
    model, identity, encoder = _runtime(args)
    root.mkdir(parents=True)

    def score(sequences):
        features = encoder.encode(sequences, batch_size=args.encoder_batch_size)
        return predict_features(
            model,
            [torch.from_numpy(x) for x in features],
            batch_size=args.batch_size,
            padding_width=args.padding_width,
        )

    for i, protein in enumerate(proteins):
        # User-supplied FASTA IDs never become filesystem paths.
        dest = root / f"protein_{i+1:04d}"
        dest.mkdir()
        result = explain(
            protein.sequence,
            score,
            operator=args.operator,
            batch_size=args.encoder_batch_size,
            progress=lambda done, total: write_json(
                dest / "progress.json", dict(done=done, total=total)
            ),
        )
        aggregate = result["aggregate"]
        _table(
            dest / "residue_scores.csv",
            [
                "position",
                "amino_acid",
                "signed_score",
                "absolute_sensitivity",
                "coverage",
            ],
            [
                dict(
                    position=j + 1,
                    amino_acid=aa,
                    signed_score=aggregate["signed_support"][j],
                    absolute_sensitivity=aggregate["absolute_sensitivity"][j],
                    coverage=aggregate["coverage"][j],
                )
                for j, aa in enumerate(protein.sequence)
            ],
        )
        _table(
            dest / "window_scores.csv",
            list(aggregate["window_means"][0]),
            aggregate["window_means"],
        )
        write_json(
            dest / "raw_scores.json",
            dict(
                original_logit=result["original_logit"],
                raw_logits=result["raw_logits"],
                ledger=result["ledger"],
            ),
        )
        write_json(
            dest / "metadata.json",
            dict(
                id=protein.id,
                sequence=protein.sequence,
                sequence_sha256=protein.sequence_sha256,
                original_length=protein.original_length,
                truncated=protein.truncated,
                model=identity,
                encoder=encoder.identity,
                protocol=result["protocol"],
                padding_width=args.padding_width,
                encoder_batch_size=args.encoder_batch_size,
                classifier_batch_size=args.batch_size,
                annotation_used=False,
            ),
        )
    write_json(
        root / "COMPLETE.json",
        dict(status="PASS", proteins=len(proteins), operator=args.operator),
    )


def train(args):
    config, model = read_config(args.config)
    rows, expected_features = read_registered_main(args.data_dir)
    store = FeatureStore(args.features_dir, expected=expected_features)
    # Only train/validation datasets enter the trainer. No test prediction here.
    fit(
        ResidueDataset(rows["train"], store, "train"),
        ResidueDataset(rows["val"], store, "val"),
        args.output,
        config,
        model,
        seed=args.seed,
        device=args.device,
    )


def evaluate(args):
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    # A registered model or completed selection lock is loaded BEFORE test data.
    if args.checkpoint:
        model, identity = load_run_model(args.checkpoint, args.device)
    else:
        model, identity = load_registered_model(
            args.models, seed=args.seed, device=args.device
        )
    rows, expected_features = read_registered_main(args.data_dir)
    store = FeatureStore(args.features_dir, expected=expected_features)
    if args.checkpoint:
        source = json.loads(
            (Path(args.checkpoint).parent / "INPUT_IDENTITY.json").read_text()
        )
        for split in ("train", "val"):
            if ResidueDataset(rows[split], store, split).identity != source[split]:
                raise ValueError(
                    "Evaluation data differs from the locked training/validation inputs"
                )
    dataset = ResidueDataset(rows[args.split], store, args.split)
    ids, y, z, _ = predict_dataset(model, dataset, 16, args.device)
    output.mkdir(parents=True)
    save_predictions(output / "predictions.csv", ids, y, z)
    write_json(
        output / "METRICS.json",
        dict(
            metrics=binary_metrics(y, z),
            model=identity,
            input=dataset.identity,
            split=args.split,
            batch_size=16,
            padding="longest_in_batch",
            threshold=0.5,
        ),
    )


def download(args):
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    names = manifest.get("groups", {}).get(args.assets, [args.assets])
    for name in names:
        entry = manifest["assets"][name]
        archive = fetch_asset(
            name, manifest, root / "downloads", offline_dir=args.offline_dir
        )
        dest = root / entry["destination"]
        if root.resolve() not in dest.resolve().parents:
            raise ValueError("Unsafe asset destination")
        if entry.get("archive"):
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                unpacked = extract_zip(archive, Path(temporary) / "unpacked")
                for source in sorted(unpacked.rglob("*")):
                    if not source.is_file():
                        continue
                    relative = source.relative_to(unpacked)
                    if str(relative) == "MEMBER_CHECKSUMS.json":
                        relative = Path(entry.get("member_manifest", str(relative)))
                    target = dest / relative
                    if root.resolve() not in target.resolve().parents:
                        raise ValueError("Unsafe existing destination link")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if file_sha256(target) != file_sha256(source):
                            raise ValueError("Existing asset differs: " + str(target))
                    else:
                        with source.open("rb") as inp, target.open("xb") as out:
                            shutil.copyfileobj(inp, out)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if file_sha256(dest) != entry["sha256"]:
                    raise ValueError("Existing asset differs")
            else:
                with archive.open("rb") as inp, dest.open("xb") as out:
                    shutil.copyfileobj(inp, out)
        print(f"Verified {name}: {dest}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="HiAntitoxinPred Main prediction, training and MAIA"
    )
    parser.add_argument("--threads", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, action in [("predict", predict), ("maia", maia)]:
        p = commands.add_parser(name)
        p.set_defaults(action=action)
        p.add_argument("--fasta", required=True)
        p.add_argument("--models", default="assets/models")
        p.add_argument("--encoder", default="assets/encoder")
        p.add_argument("--cache")
        p.add_argument("--seed", type=int)
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--encoder-batch-size", type=int, default=2)
        p.add_argument("--output", required=True)
        if name == "maia":
            p.add_argument(
                "--padding-width",
                type=int,
                default=200,
                help="Classifier padding width; use archived width for reference replay (default: 200)",
            )
            p.add_argument(
                "--operator", choices=["U19", "SHUFFLE_COMPOSITION"], default="U19"
            )
    p = commands.add_parser("train")
    p.set_defaults(action=train)
    p.add_argument("--config", default="configs/main.json")
    p.add_argument("--data-dir", default="data/main")
    p.add_argument("--features-dir", default="assets/main_features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.set_defaults(action=evaluate)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--models")
    p.add_argument("--seed", type=int)
    p.add_argument("--data-dir", default="data/main")
    p.add_argument("--features-dir", default="assets/main_features")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p = commands.add_parser("download")
    p.set_defaults(action=download)
    p.add_argument("--assets", required=True)
    p.add_argument("--manifest", default="pretrained/manifest.json")
    p.add_argument("--directory", default="assets")
    p.add_argument("--offline-dir")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    return args.action(args)
