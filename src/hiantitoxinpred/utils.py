from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(data: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, ensure_ascii=False, indent=2)
    return path


def read_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(data, handle)
    return path


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        remain = int(seconds % 60)
        return f"{minutes}m{remain:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes:02d}m"


def resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device)
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if ":" in requested:
            try:
                gpu_idx = int(requested.split(":", 1)[1])
            except ValueError:
                gpu_idx = 0
            if gpu_idx >= torch.cuda.device_count():
                return torch.device("cpu")
    return torch.device(requested)


def set_global_seed(seed: Optional[int], deterministic: bool = False) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def normalize_protein_sequence(sequence: str) -> str:
    sequence = str(sequence).strip().upper()
    return "".join("X" if aa in {"U", "Z", "O", "B"} else aa for aa in sequence)


def spaced_sequence(sequence: str) -> str:
    return " ".join(list(normalize_protein_sequence(sequence)))


def truncate_sequence(sequence: str, max_length: Optional[int] = 100) -> str:
    sequence = normalize_protein_sequence(sequence)
    if max_length is None:
        return sequence
    return sequence[: int(max_length)]


def load_sequences_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path)
        if {"id", "sequence", "label"}.issubset(df.columns):
            out = df[["id", "sequence", "label"]].copy()
            out["id"] = out["id"].astype(str)
            out["sequence"] = out["sequence"].astype(str)
            out["label"] = out["label"].astype(int)
            return out
    except Exception:
        pass

    df = pd.read_csv(path, header=None, names=["id", "sequence", "label"])
    if df.iloc[0].astype(str).tolist() == ["id", "sequence", "label"]:
        df = df.iloc[1:].reset_index(drop=True)
    out = df[["id", "sequence", "label"]].copy()
    out["id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def write_sequences_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def which_any(candidates: Iterable[str]) -> Optional[str]:
    executable_dir = Path(sys.executable).resolve().parent
    for candidate in candidates:
        local_candidate = executable_dir / candidate
        if local_candidate.exists() and os.access(local_candidate, os.X_OK):
            return str(local_candidate)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def flatten_confusion_matrix(confusion_matrix: List[List[int]]) -> Dict[str, int]:
    if not confusion_matrix or len(confusion_matrix) != 2 or len(confusion_matrix[0]) != 2 or len(confusion_matrix[1]) != 2:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    return {
        "tn": int(confusion_matrix[0][0]),
        "fp": int(confusion_matrix[0][1]),
        "fn": int(confusion_matrix[1][0]),
        "tp": int(confusion_matrix[1][1]),
    }
