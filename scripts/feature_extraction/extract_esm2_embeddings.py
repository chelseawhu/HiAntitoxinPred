#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hiantitoxinpred.data import save_feature_records  # noqa: E402
from hiantitoxinpred.utils import load_sequences_csv, normalize_protein_sequence, resolve_device  # noqa: E402


def model_ref_from_config(value: str | Path) -> str:
    raw_value = str(value)
    raw_path = Path(raw_value)
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)
    candidate = (REPO_ROOT / raw_path).resolve()
    if candidate.exists():
        return str(candidate)
    return raw_value

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def discover_dataset_csvs(dataset_root: Path) -> List[Path]:
    csvs = sorted(dataset_root.rglob("processed_sequences.csv"))
    if not csvs:
        raise FileNotFoundError(f"No processed_sequences.csv files found under {dataset_root}")
    return csvs


def dataset_name_from_csv(dataset_root: Path, csv_path: Path) -> str:
    return str(csv_path.parent.relative_to(dataset_root))


def normalize_sequence(value: Any) -> str:
    return "".join(str(value).strip().upper().split())


def align_hidden_to_sequence(hidden: np.ndarray, expected_length: int) -> np.ndarray:
    if hidden.shape[0] == expected_length:
        return hidden.astype(np.float32, copy=False)
    if hidden.shape[0] > expected_length:
        return hidden[:expected_length].astype(np.float32, copy=False)
    pad = np.zeros((expected_length - hidden.shape[0], hidden.shape[1]), dtype=np.float32)
    return np.concatenate([hidden.astype(np.float32, copy=False), pad], axis=0)


def extract_esm2_per_residue_flexible(
    *,
    sequences_csv: Path,
    model_ref: str,
    output_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
    description: str,
) -> Dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    sequences_df = load_sequences_csv(sequences_csv)
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModel.from_pretrained(model_ref)
    model = model.to(resolve_device(device))
    model.eval()

    ids = sequences_df["id"].astype(str).tolist()
    labels = sequences_df["label"].astype(int).tolist()
    sequences = [normalize_protein_sequence(seq) for seq in sequences_df["sequence"].astype(str).tolist()]
    special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids)), dtype=torch.long)
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc=description, unit="batch"):
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
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
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
                records.append(
                    {
                        "id": ids[start + offset],
                        "label": int(labels[start + offset]),
                        "sequence": batch_sequences[offset],
                        "features": per_residue,
                        "length": int(per_residue.shape[0]),
                        "feature_dim": int(per_residue.shape[1]),
                        "feature_type": "per_residue",
                    }
                )

    save_feature_records(output_path, records, encoder="esm2", feature_type="per_residue")
    return {
        "encoder": "esm2",
        "count": len(records),
        "feature_dim": int(records[0]["feature_dim"]) if records else 0,
        "output_path": str(output_path.resolve()),
    }


def collect_unique_sequences(config: Dict[str, Any], *, skip_completed: bool = False) -> Dict[str, Any]:
    dataset_root = resolve_path(config["dataset_root"])
    global_dir = resolve_path(config["global_feature_dir"])
    global_dir.mkdir(parents=True, exist_ok=True)

    unique_csv = global_dir / "all_unique_sequences.csv"
    unique_fasta = global_dir / "all_unique_sequences.fasta"
    membership_csv = global_dir / "dataset_membership.csv"
    conflict_csv = global_dir / "sequence_id_conflicts.csv"
    summary_json = global_dir / "unique_sequence_summary.json"

    if skip_completed and unique_csv.exists() and membership_csv.exists() and summary_json.exists():
        print(f"[skip] unique sequence table exists: {unique_csv}")
        return read_json(summary_json)

    id_col = str(config.get("id_column", "id"))
    seq_col = str(config.get("sequence_column", "sequence"))
    label_col = str(config.get("label_column", "label"))

    rows_by_id: Dict[str, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []
    memberships: List[Dict[str, Any]] = []

    dataset_csvs = discover_dataset_csvs(dataset_root)
    for csv_path in dataset_csvs:
        dataset_name = dataset_name_from_csv(dataset_root, csv_path)
        df = pd.read_csv(csv_path)
        missing = {id_col, seq_col, label_col} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for row in df.itertuples(index=False):
            item = row._asdict()
            sample_id = str(item[id_col])
            sequence = normalize_sequence(item[seq_col])
            label = int(item[label_col])
            if not sample_id or not sequence:
                raise ValueError(f"{csv_path} contains an empty id or sequence")

            memberships.append(
                {
                    "dataset_name": dataset_name,
                    "id": sample_id,
                    "label": label,
                    "sequence_length": len(sequence),
                }
            )

            previous = rows_by_id.get(sample_id)
            if previous is None:
                rows_by_id[sample_id] = {
                    "id": sample_id,
                    "label": label,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "source_dataset_count": 1,
                }
                continue

            previous["source_dataset_count"] += 1
            if previous["sequence"] != sequence or int(previous["label"]) != label:
                conflicts.append(
                    {
                        "id": sample_id,
                        "previous_label": int(previous["label"]),
                        "new_label": label,
                        "previous_length": len(previous["sequence"]),
                        "new_length": len(sequence),
                        "dataset_name": dataset_name,
                    }
                )

    if conflicts:
        pd.DataFrame(conflicts).to_csv(conflict_csv, index=False)
        raise ValueError(f"Found {len(conflicts)} ID conflicts. See {conflict_csv}")

    unique_df = pd.DataFrame(rows_by_id.values()).sort_values("id").reset_index(drop=True)
    membership_df = pd.DataFrame(memberships).sort_values(["dataset_name", "id"]).reset_index(drop=True)
    unique_df.to_csv(unique_csv, index=False)
    membership_df.to_csv(membership_csv, index=False)

    with unique_fasta.open("w", encoding="utf-8") as handle:
        for item in unique_df.itertuples(index=False):
            handle.write(f">{item.id}|label={int(item.label)}|len={int(item.sequence_length)}\n")
            seq = str(item.sequence)
            for start in range(0, len(seq), 80):
                handle.write(seq[start : start + 80] + "\n")

    summary = {
        "dataset_csv_count": len(dataset_csvs),
        "dataset_row_total": int(len(membership_df)),
        "unique_sequence_count": int(len(unique_df)),
        "positive_count": int((unique_df["label"] == 1).sum()),
        "negative_count": int((unique_df["label"] == 0).sum()),
        "min_length": int(unique_df["sequence_length"].min()),
        "max_length": int(unique_df["sequence_length"].max()),
        "mean_length": float(unique_df["sequence_length"].mean()),
        "unique_csv": str(unique_csv),
        "unique_fasta": str(unique_fasta),
        "membership_csv": str(membership_csv),
    }
    write_json(summary, summary_json)
    print(f"[prepare_unique] datasets={len(dataset_csvs)} rows={len(membership_df)} unique={len(unique_df)}")
    print(f"[prepare_unique] wrote {unique_csv}")
    return summary


def extract_global_features(config: Dict[str, Any], *, device: str | None, batch_size: int | None, skip_completed: bool) -> Dict[str, Any]:
    global_dir = resolve_path(config["global_feature_dir"])
    unique_csv = global_dir / "all_unique_sequences.csv"
    output_path = global_dir / "esm2_per_residue_features.pkl"
    info_json = global_dir / "esm2_feature_extraction_info.json"

    if not unique_csv.exists():
        raise FileNotFoundError(f"Run prepare_unique first; missing {unique_csv}")
    if skip_completed and output_path.exists() and info_json.exists():
        print(f"[skip] global ESM2 features exist: {output_path}")
        return read_json(info_json)

    model_ref = model_ref_from_config(config["esm2_model_path"])
    info = extract_esm2_per_residue_flexible(
        sequences_csv=unique_csv,
        model_ref=model_ref,
        output_path=output_path,
        device=device or str(config.get("device", "cuda:0")),
        batch_size=int(batch_size or config.get("batch_size", 2)),
        max_length=int(config.get("max_length", 256)),
        description=str(config.get("progress_description", "Extracting ESM2")),
    )
    info["sequences_csv"] = str(unique_csv)
    info["model_path"] = str(model_ref)
    info["max_length"] = int(config.get("max_length", 256))
    info["batch_size"] = int(batch_size or config.get("batch_size", 2))
    info["device"] = device or str(config.get("device", "cuda:0"))
    write_json(info, info_json)
    print(f"[extract] saved {output_path}")
    return info


def verify_and_write_dataset_maps(config: Dict[str, Any]) -> Dict[str, Any]:
    from hiantitoxinpred.data import load_feature_records

    dataset_root = resolve_path(config["dataset_root"])
    global_dir = resolve_path(config["global_feature_dir"])
    map_dir = resolve_path(config["dataset_feature_map_dir"])
    map_dir.mkdir(parents=True, exist_ok=True)

    feature_path = global_dir / "esm2_per_residue_features.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"Run extract first; missing {feature_path}")

    records = load_feature_records(feature_path)
    feature_map = {item["id"]: item for item in records}
    expected_dim = int(config.get("expected_feature_dim", 1280))

    inventory_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    length_mismatch_rows: List[Dict[str, Any]] = []
    dataset_maps: Dict[str, Any] = {}

    for csv_path in discover_dataset_csvs(dataset_root):
        dataset_name = dataset_name_from_csv(dataset_root, csv_path)
        df = pd.read_csv(csv_path)
        ids = df["id"].astype(str).tolist()
        lengths = df["sequence"].astype(str).map(lambda value: len(normalize_sequence(value))).tolist()
        labels = df["label"].astype(int).tolist()

        for sample_id, expected_length, label in zip(ids, lengths, labels):
            item = feature_map.get(sample_id)
            if item is None:
                missing_rows.append({"dataset_name": dataset_name, "id": sample_id})
                continue
            actual_length = int(item["features"].shape[0])
            feature_dim = int(item["features"].shape[1])
            if actual_length != int(expected_length) or feature_dim != expected_dim:
                length_mismatch_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "id": sample_id,
                        "expected_length": int(expected_length),
                        "actual_length": actual_length,
                        "feature_dim": feature_dim,
                    }
                )

        safe_name = dataset_name.replace("/", "__")
        map_path = map_dir / f"{safe_name}.json"
        payload = {
            "dataset_name": dataset_name,
            "processed_sequences_csv": str(csv_path),
            "global_feature_path": str(feature_path),
            "sample_count": int(len(ids)),
            "positive_count": int(sum(labels)),
            "negative_count": int(len(labels) - sum(labels)),
            "ids": ids,
        }
        write_json(payload, map_path)
        dataset_maps[dataset_name] = str(map_path)
        inventory_rows.append(
            {
                "dataset_name": dataset_name,
                "sample_count": int(len(ids)),
                "positive_count": int(sum(labels)),
                "negative_count": int(len(labels) - sum(labels)),
                "map_path": str(map_path),
            }
        )

    inventory_csv = map_dir / "dataset_feature_inventory.csv"
    pd.DataFrame(inventory_rows).sort_values("dataset_name").to_csv(inventory_csv, index=False)

    missing_csv = map_dir / "missing_feature_records.csv"
    mismatch_csv = map_dir / "feature_length_mismatches.csv"
    pd.DataFrame(missing_rows).to_csv(missing_csv, index=False)
    pd.DataFrame(length_mismatch_rows).to_csv(mismatch_csv, index=False)

    summary = {
        "global_feature_path": str(feature_path),
        "feature_record_count": int(len(records)),
        "dataset_count": int(len(inventory_rows)),
        "missing_count": int(len(missing_rows)),
        "length_mismatch_count": int(len(length_mismatch_rows)),
        "inventory_csv": str(inventory_csv),
        "missing_csv": str(missing_csv),
        "mismatch_csv": str(mismatch_csv),
        "dataset_maps": dataset_maps,
    }
    write_json(summary, map_dir / "dataset_feature_map_summary.json")

    if missing_rows or length_mismatch_rows:
        raise ValueError(
            "Feature verification failed: "
            f"missing={len(missing_rows)}, length_mismatch={len(length_mismatch_rows)}"
        )

    print(f"[verify] datasets={len(inventory_rows)} feature_records={len(records)}")
    print(f"[verify] wrote {inventory_csv}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare global ESM2 per-residue features for HiAntitoxinPred datasets")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["prepare_unique", "extract", "verify", "all"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = read_json(config_path)
    stages = ["prepare_unique", "extract", "verify"] if "all" in args.stages else args.stages

    print(f"Config: {config_path}")
    for stage in stages:
        print(f"\n== Stage: {stage} ==")
        if stage == "prepare_unique":
            collect_unique_sequences(config, skip_completed=args.skip_completed)
        elif stage == "extract":
            extract_global_features(config, device=args.device, batch_size=args.batch_size, skip_completed=args.skip_completed)
        elif stage == "verify":
            verify_and_write_dataset_maps(config)
        else:
            raise ValueError(f"Unsupported stage: {stage}")


if __name__ == "__main__":
    main()
