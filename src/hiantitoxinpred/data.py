from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import ensure_dir, load_sequences_csv, read_json, read_pickle, write_json


def _coerce_record_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    elif isinstance(payload, dict):
        if all(isinstance(value, dict) for value in payload.values()):
            payload = list(payload.values())
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Unsupported feature payload format")
    return [item for item in payload if isinstance(item, dict)]


def load_feature_records(path: str | Path) -> List[Dict[str, Any]]:
    records = _coerce_record_list(read_pickle(path))
    output: List[Dict[str, Any]] = []
    for item in records:
        if "id" not in item:
            continue
        features = item.get("features", item.get("embedding"))
        if features is None:
            continue
        features_np = np.asarray(features, dtype=np.float32)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.ndim != 2:
            continue
        output.append(
            {
                "id": str(item["id"]),
                "label": int(item.get("label", 0)),
                "features": features_np,
                "length": int(item.get("length", features_np.shape[0])),
                "feature_dim": int(item.get("feature_dim", features_np.shape[1])),
                "sequence": item.get("sequence"),
            }
        )
    if not output:
        raise ValueError(f"No usable feature records found in {path}")
    return output


def load_embedding_records(path: str | Path) -> Dict[str, np.ndarray]:
    records = _coerce_record_list(read_pickle(path))
    embedding_map: Dict[str, np.ndarray] = {}
    for item in records:
        sample_id = item.get("id")
        embedding = item.get("embedding", item.get("features"))
        if sample_id is None or embedding is None:
            continue
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        embedding_map[str(sample_id)] = vector
    if not embedding_map:
        raise ValueError(f"No usable embedding records found in {path}")
    return embedding_map


def save_feature_records(
    path: str | Path,
    records: List[Dict[str, Any]],
    *,
    encoder: str,
    feature_type: str = "per_residue",
) -> Path:
    from .utils import write_pickle

    payload = {
        "encoder": encoder,
        "feature_type": feature_type,
        "count": len(records),
        "records": records,
    }
    return write_pickle(payload, path)


class ESMOnlyDataset(Dataset):
    def __init__(self, esm_features: Sequence[np.ndarray], labels: Sequence[float], ids: Sequence[str]) -> None:
        self.esm_list = [np.asarray(item, dtype=np.float32) for item in esm_features]
        self.labels = torch.as_tensor(np.asarray(labels, dtype=np.float32).reshape(-1, 1))
        self.ids = list(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        esm = self.esm_list[index]
        return {
            "esm": torch.as_tensor(esm, dtype=torch.float32),
            "esm_len": int(esm.shape[0]),
            "one_hot": torch.zeros(1, dtype=torch.float32),
            "aaindex": torch.zeros(1, dtype=torch.float32),
            "label": self.labels[index],
            "id": self.ids[index],
        }


def window_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [item["esm_len"] for item in batch]
    max_len = max(lengths) if lengths else 0
    feat_dim = batch[0]["esm"].shape[1] if batch else 0
    padded = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    for idx, item in enumerate(batch):
        cur_len = item["esm_len"]
        padded[idx, :cur_len, :] = item["esm"]
    return {
        "esm": padded,
        "esm_len": torch.as_tensor(lengths, dtype=torch.long),
        "one_hot": torch.stack([item["one_hot"] for item in batch], dim=0),
        "aaindex": torch.stack([item["aaindex"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "id": [item["id"] for item in batch],
    }


def dataset_stats(dataset: ESMOnlyDataset) -> Dict[str, Any]:
    labels = dataset.labels.detach().cpu().numpy().reshape(-1).astype(int)
    return {
        "size": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
    }


def dataset_records(dataset: ESMOnlyDataset) -> List[Dict[str, Any]]:
    labels = dataset.labels.detach().cpu().numpy().reshape(-1).astype(int)
    return [{"id": sample_id, "label": int(label)} for sample_id, label in zip(dataset.ids, labels)]


def validate_split_manifest(manifest: Dict[str, Any]) -> None:
    train_ids = set(manifest["ids_train"])
    val_ids = set(manifest["ids_val"])
    test_ids = set(manifest["ids_test"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Split manifest contains overlapping ids")


def build_split_manifest(
    sequences_csv: str | Path,
    feature_records: Sequence[Dict[str, Any]],
    output_path: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Dict[str, Any]:
    sequences_df = load_sequences_csv(sequences_csv)
    label_map = dict(zip(sequences_df["id"], sequences_df["label"]))
    ids = [item["id"] for item in feature_records]
    labels = [int(label_map.get(sample_id, item["label"])) for sample_id, item in zip(ids, feature_records)]
    ids_temp, ids_test, y_temp, _ = train_test_split(
        ids,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    ids_train, ids_val, _, _ = train_test_split(
        ids_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp if len(set(y_temp)) > 1 else None,
    )
    manifest = {
        "random_state": int(random_state),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "ids_train": list(ids_train),
        "ids_val": list(ids_val),
        "ids_test": list(ids_test),
    }
    validate_split_manifest(manifest)
    write_json(manifest, output_path)
    return manifest


def load_or_create_split_manifest(
    feature_records: Sequence[Dict[str, Any]],
    sequences_csv: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[Dict[str, Any], Path]:
    output_dir = ensure_dir(output_dir)
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        validate_split_manifest(manifest)
        return manifest, manifest_path
    manifest = build_split_manifest(
        sequences_csv=sequences_csv,
        feature_records=feature_records,
        output_path=manifest_path,
        random_state=random_state,
        test_size=test_size,
        val_size=val_size,
    )
    return manifest, manifest_path


def build_datasets_from_manifest(
    feature_records: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Tuple[ESMOnlyDataset, ESMOnlyDataset, ESMOnlyDataset, Dict[str, Any]]:
    validate_split_manifest(manifest)
    feature_map = {item["id"]: item for item in feature_records}

    def gather(ids: Sequence[str]) -> Tuple[List[np.ndarray], List[int]]:
        rows = []
        labels = []
        for sample_id in ids:
            if sample_id not in feature_map:
                raise ValueError(f"Missing feature record for id={sample_id}")
            row = feature_map[sample_id]
            rows.append(np.asarray(row["features"], dtype=np.float32))
            labels.append(int(row["label"]))
        return rows, labels

    train_esm, train_labels = gather(manifest["ids_train"])
    val_esm, val_labels = gather(manifest["ids_val"])
    test_esm, test_labels = gather(manifest["ids_test"])

    train_ds = ESMOnlyDataset(train_esm, train_labels, manifest["ids_train"])
    val_ds = ESMOnlyDataset(val_esm, val_labels, manifest["ids_val"])
    test_ds = ESMOnlyDataset(test_esm, test_labels, manifest["ids_test"])
    esm_dim = int(train_esm[0].shape[1]) if train_esm else int(val_esm[0].shape[1])
    meta = {
        "esm_dim": esm_dim,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "ids_train": list(train_ds.ids),
        "ids_val": list(val_ds.ids),
        "ids_test": list(test_ds.ids),
    }
    return train_ds, val_ds, test_ds, meta


def export_split_fastas(
    manifest: Dict[str, Any],
    sequences_csv: str | Path,
    output_dir: str | Path,
) -> Dict[str, str]:
    sequences_df = load_sequences_csv(sequences_csv)
    seq_map = dict(zip(sequences_df["id"], sequences_df["sequence"]))
    output_dir = ensure_dir(output_dir)
    outputs: Dict[str, str] = {}

    def write_fasta(split_name: str, ids: Sequence[str], label: Optional[int] = None) -> Path:
        fasta_path = output_dir / f"{split_name}.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            for sample_id in ids:
                sequence = seq_map.get(sample_id)
                if sequence is None:
                    continue
                header = sample_id if label is None else f"{sample_id}|label={label}"
                handle.write(f">{header}\n{sequence}\n")
        return fasta_path

    for split_name, key in [("train", "ids_train"), ("val", "ids_val"), ("test", "ids_test")]:
        ids = list(manifest[key])
        outputs[f"{split_name}_all"] = str(write_fasta(f"{split_name}_all", ids))
        pos_ids = [sample_id for sample_id in ids if int(sequences_df.loc[sequences_df["id"] == sample_id, "label"].iloc[0]) == 1]
        neg_ids = [sample_id for sample_id in ids if int(sequences_df.loc[sequences_df["id"] == sample_id, "label"].iloc[0]) == 0]
        outputs[f"{split_name}_positive"] = str(write_fasta(f"{split_name}_positive", pos_ids, label=1))
        outputs[f"{split_name}_negative"] = str(write_fasta(f"{split_name}_negative", neg_ids, label=0))

    write_json(outputs, output_dir / "fasta_inventory.json")
    return outputs


def load_relabeled_positive_table(path: str | Path) -> pd.DataFrame:
    df = load_sequences_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Relabeled table is missing label column: {path}")
    return df


def group_relabeled_positive_records_by_label(feature_records: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for item in feature_records:
        label = int(item["label"])
        groups.setdefault(label, []).append(item)
    return groups
