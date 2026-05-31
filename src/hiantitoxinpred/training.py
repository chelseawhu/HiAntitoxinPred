from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    build_datasets_from_manifest,
    dataset_records,
    dataset_stats,
    load_feature_records,
    window_collate_fn,
)
from .metrics import apply_fixed_threshold, compute_binary_classification_metrics
from .utils import ensure_dir, resolve_device, seed_worker, set_global_seed, write_json

from .model import LabelSmoothingLoss, SerialCLSRescueNet, TrackAMinimalRescueNet, TrackBSecondGenRescueNet


@dataclass
class CNNRescueTrainingConfig:
    esm_dim: int
    one_hot_dim: int = 1
    aaindex_dim: int = 1
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.2
    use_onehot: bool = False
    use_aaindex: bool = False
    use_cnn: bool = True
    use_transformer: bool = True
    batch_size: int = 16
    epochs: int = 150
    learning_rate: float = 2.3e-5
    patience: int = 0
    min_delta: float = 1e-5
    random_seed: Optional[int] = None
    deterministic: bool = False
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    threshold_policy: str = "fixed_prob"
    fixed_threshold_prob: Optional[float] = 0.5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    pos_weight_cap: float = 1.5
    model_family: str = "track_a_minimal_rescue"
    battlefield: str = "main1280"
    evaluation_protocols: List[str] = field(default_factory=lambda: ["fixed_0p5", "shared_validation_selected_threshold"])
    cnn_branch_mode: str = "independent_sequence_branch"
    cnn_pooling_mode: str = "topk"
    cnn_branch_gate_mode: str = "sample_adaptive_branch_softmax"
    training_schedule: Dict[str, Any] = field(default_factory=lambda: {"type": "single_stage"})
    top_k: int = 3
    cnn_scalar_gate_init_prob: float = 0.1
    main_pooling_mode: str = "dynamic"
    cnn_only_mode: bool = False
    checkpoint_selection_metric: str = "validation_max_f1"

    def __post_init__(self) -> None:
        if self.pin_memory is None:
            self.pin_memory = torch.cuda.is_available()
        self.threshold_policy = str(self.threshold_policy).strip().lower()
        if self.threshold_policy not in {"validation_f1", "validation_max_f1", "fixed_prob"}:
            raise ValueError(f"Unsupported threshold_policy: {self.threshold_policy}")
        if self.threshold_policy == "fixed_prob":
            if self.fixed_threshold_prob is None:
                raise ValueError("fixed_threshold_prob must be provided when threshold_policy='fixed_prob'")
            if not 0.0 < float(self.fixed_threshold_prob) < 1.0:
                raise ValueError("fixed_threshold_prob must be strictly between 0 and 1")
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if float(self.pos_weight_cap) <= 0.0:
            raise ValueError("pos_weight_cap must be positive")
        if self.model_family not in {"track_a_minimal_rescue", "track_b_second_gen_rescue", "serial_cls_rescue"}:
            raise ValueError(f"Unsupported model_family: {self.model_family}")
        if self.cnn_pooling_mode != "topk":
            raise ValueError("cnn_pooling_mode must be 'topk'")
        self.main_pooling_mode = str(self.main_pooling_mode).strip().lower()
        if self.main_pooling_mode not in {"dynamic", "mean", "max"}:
            raise ValueError("main_pooling_mode must be 'dynamic', 'mean', or 'max'")
        if not 0.0 < float(self.cnn_scalar_gate_init_prob) < 1.0:
            raise ValueError("cnn_scalar_gate_init_prob must be strictly between 0 and 1")
        self.checkpoint_selection_metric = str(self.checkpoint_selection_metric).strip().lower()
        if self.checkpoint_selection_metric not in {"validation_max_f1", "validation_loss"}:
            raise ValueError(
                "checkpoint_selection_metric must be 'validation_max_f1' or 'validation_loss'"
            )


def probability_to_logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return float(np.log(probability / (1.0 - probability)))


def logit_to_probability(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(logit))))


def create_dataloader(
    dataset,
    config: CNNRescueTrainingConfig,
    *,
    shuffle: bool = True,
    collate_fn=None,
    seed: Optional[int] = None,
) -> DataLoader:
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = seed_worker
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=bool(config.pin_memory),
        drop_last=shuffle,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def plot_training_curves(logs: Dict[str, Any], save_path: str | Path, title: str) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(logs["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    axes[0].plot(epochs, logs["train_loss"], label="train_loss")
    if logs["val_loss"]:
        axes[0].plot(epochs[: len(logs["val_loss"])], logs["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    if logs["val_f1"]:
        axes[1].plot(epochs[: len(logs["val_f1"])], logs["val_f1"], label="val_f1")
    axes[1].set_title("Validation F1")
    axes[1].set_ylim(0, 1)
    if logs["val_mcc"]:
        axes[2].plot(epochs[: len(logs["val_mcc"])], logs["val_mcc"], label="val_mcc")
    axes[2].set_title("Validation MCC")
    axes[2].set_ylim(-0.1, 1.0)
    if logs["lr"]:
        axes[3].plot(epochs[: len(logs["lr"])], logs["lr"], label="lr")
        axes[3].set_yscale("log")
    axes[3].set_title("Learning Rate")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_cnn_rescue_model(config: CNNRescueTrainingConfig, device: str | torch.device) -> nn.Module:
    kwargs = {
        "esm_dim": config.esm_dim,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "one_hot_dim": config.one_hot_dim,
        "aaindex_dim": config.aaindex_dim,
        "use_onehot": config.use_onehot,
        "use_aaindex": config.use_aaindex,
        "use_cnn": config.use_cnn,
        "use_transformer": config.use_transformer,
        "top_k": config.top_k,
        "main_pooling_mode": config.main_pooling_mode,
    }
    if config.model_family == "track_a_minimal_rescue":
        model = TrackAMinimalRescueNet(**kwargs)
    elif config.model_family == "serial_cls_rescue":
        model = SerialCLSRescueNet(**kwargs)
    else:
        model = TrackBSecondGenRescueNet(
            **kwargs,
            scalar_gate_init_prob=float(config.cnn_scalar_gate_init_prob),
            cnn_only_mode=bool(config.cnn_only_mode),
        )
    return model.to(resolve_device(device))


def _compute_fixed_metrics(labels: np.ndarray, logits: np.ndarray, threshold_logit: float) -> Dict[str, Any]:
    metrics = apply_fixed_threshold(labels, logits, threshold_logit, score_kind="logit")
    metrics["auc"] = metrics["auc_roc"]
    return metrics


def _select_validation_threshold_by_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_row = None
    for threshold in np.arange(0.05, 0.951, 0.01):
        metrics = compute_binary_classification_metrics(labels, probs, threshold, score_kind="prob")
        row = (
            float(metrics["f1"]),
            float(metrics["mcc"]),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        if best_row is None or row > best_row:
            best_row = row
    return probability_to_logit(best_row[-1] if best_row is not None else 0.5)


def _aggregate_diagnostics(
    *,
    branch_names: List[str],
    gate_values: List[np.ndarray],
    branch_gates: List[np.ndarray],
    topk_scores: List[np.ndarray],
    topk_positions: List[np.ndarray],
) -> Dict[str, Any]:
    if not gate_values:
        return {
            "cnn_gate_mean": None,
            "cnn_gate_std": None,
            "branch_gate_vector": None,
            "topk_motif_scores_summary": None,
            "topk_positions_summary": None,
        }
    gate_arr = np.concatenate(gate_values, axis=0)
    branch_arr = np.concatenate(branch_gates, axis=0) if branch_gates else None
    score_arr = np.concatenate(topk_scores, axis=0) if topk_scores else None
    position_arr = np.concatenate(topk_positions, axis=0) if topk_positions else None
    score_summary = None
    position_summary = None
    branch_vector = None
    if branch_arr is not None:
        branch_vector = {name: float(branch_arr[:, idx].mean()) for idx, name in enumerate(branch_names)}
    if score_arr is not None:
        score_summary = {name: float(score_arr[:, idx].mean()) for idx, name in enumerate(branch_names)}
    if position_arr is not None:
        position_summary = {name: float(position_arr[:, idx].mean()) for idx, name in enumerate(branch_names)}
    return {
        "cnn_gate_mean": float(gate_arr.mean()),
        "cnn_gate_std": float(gate_arr.std(ddof=0)),
        "branch_gate_vector": branch_vector,
        "topk_motif_scores_summary": score_summary,
        "topk_positions_summary": position_summary,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device,
    *,
    fixed_threshold: Optional[float] = None,
    return_raw: bool = False,
    collect_diagnostics: bool = False,
) -> Dict[str, Any]:
    device = resolve_device(device)
    model.eval()
    all_logits: List[float] = []
    all_labels: List[int] = []
    gate_values: List[np.ndarray] = []
    branch_gates: List[np.ndarray] = []
    topk_scores: List[np.ndarray] = []
    topk_positions: List[np.ndarray] = []
    branch_names = list(getattr(model, "branch_names", []))

    for batch in data_loader:
        esm = batch["esm"].to(device)
        one_hot = batch["one_hot"].to(device)
        aaindex = batch["aaindex"].to(device)
        esm_len = batch["esm_len"].to(device)
        labels = batch["label"].to(device)
        payload = model.compute_outputs(
            esm,
            one_hot,
            aaindex,
            esm_len=esm_len,
            return_diagnostics=collect_diagnostics,
        )
        logits = payload["logits"]
        logits_np = np.clip(logits.detach().cpu().numpy().reshape(-1), -10.0, 10.0)
        labels_np = labels.detach().cpu().numpy().reshape(-1).astype(int)
        all_logits.extend(logits_np.tolist())
        all_labels.extend(labels_np.tolist())
        diagnostics = payload.get("diagnostics")
        if collect_diagnostics and diagnostics is not None:
            gate_values.append(diagnostics["cnn_gate"].detach().cpu().numpy().reshape(-1, 1))
            branch_gates.append(diagnostics["branch_gates"].detach().cpu().numpy())
            topk_scores.append(diagnostics["topk_score_matrix"].detach().cpu().numpy())
            topk_positions.append(diagnostics["topk_position_matrix"].detach().cpu().numpy())

    logits_arr = np.asarray(all_logits, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int32)
    if fixed_threshold is None:
        threshold_logit = _select_validation_threshold_by_f1(labels_arr, logits_arr)
    else:
        threshold_logit = float(fixed_threshold)
    metrics = _compute_fixed_metrics(labels_arr, logits_arr, threshold_logit)
    if fixed_threshold is None:
        metrics["threshold_source"] = "validation_selected"
    else:
        metrics["threshold_source"] = "fixed"
    diagnostics_payload = _aggregate_diagnostics(
        branch_names=branch_names,
        gate_values=gate_values,
        branch_gates=branch_gates,
        topk_scores=topk_scores,
        topk_positions=topk_positions,
    )
    metrics.update(diagnostics_payload)
    if return_raw:
        metrics["logits_array"] = logits_arr
        metrics["labels_array"] = labels_arr
    return metrics


def _named_requires_grad(params: List[torch.nn.Parameter], enabled: bool) -> None:
    for param in params:
        param.requires_grad = enabled


def _build_optimizer(params: List[torch.nn.Parameter], learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    if not params:
        raise ValueError("No trainable parameters were selected for the current stage")
    return torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-7,
    )


def _configure_stage(model: nn.Module, config: CNNRescueTrainingConfig, stage_name: str) -> torch.optim.Optimizer:
    groups = model.get_parameter_groups()
    all_params = []
    for params in groups.values():
        all_params.extend(params)
    _named_requires_grad(all_params, False)

    if stage_name == "single_stage":
        trainable = all_params
        lr = config.learning_rate
    elif stage_name == "stage1_main":
        trainable = groups["main"] + groups["classifier"]
        lr = config.learning_rate
    elif stage_name == "stage2_cnn":
        trainable = groups["cnn"] + groups["fusion"] + groups["aux"]
        lr = config.learning_rate
    elif stage_name == "stage3_joint":
        trainable = all_params
        lr = config.learning_rate * float(config.training_schedule.get("stage3_lr_scale", 0.25))
    else:
        raise ValueError(f"Unsupported stage_name: {stage_name}")
    _named_requires_grad(trainable, True)
    return _build_optimizer(trainable, lr, config.weight_decay)


def _select_stage_name(config: CNNRescueTrainingConfig, epoch_idx: int) -> str:
    # This final rerun intentionally returns to the original end-to-end recipe:
    # all active branches are optimized together from epoch 1.
    return "single_stage"


def train_cnn_rescue_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str | torch.device,
    config: CNNRescueTrainingConfig,
    *,
    checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> tuple[float, Dict[str, Any]]:
    device = resolve_device(device)
    labels_np = train_loader.dataset.labels.detach().cpu().numpy().reshape(-1)
    num_pos = max(float(labels_np.sum()), 1.0)
    num_neg = max(float(len(labels_np) - labels_np.sum()), 1.0)
    adjusted_pos_weight = min(np.sqrt(num_neg / num_pos), float(config.pos_weight_cap))
    criterion = LabelSmoothingLoss(smoothing=config.label_smoothing, pos_weight=adjusted_pos_weight)

    logs: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_mcc": [],
        "lr": [],
        "stage_name": [],
        "pos_weight": float(adjusted_pos_weight),
        "pos_weight_cap": float(config.pos_weight_cap),
    }
    best_state = None
    best_val_loss = float("inf")
    best_f1 = -1.0
    best_val_mcc = -1.0
    best_epoch = 0
    best_stage = ""
    epochs_since_improve = 0

    fixed_threshold = None
    if config.threshold_policy == "fixed_prob":
        fixed_threshold = probability_to_logit(float(config.fixed_threshold_prob))

    current_stage = None
    optimizer = None
    for epoch in range(config.epochs):
        stage_name = _select_stage_name(config, epoch)
        if stage_name != current_stage or optimizer is None:
            optimizer = _configure_stage(model, config, stage_name)
            current_stage = stage_name
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            esm = torch.clamp(batch["esm"].to(device, non_blocking=True), -10.0, 10.0)
            one_hot = batch["one_hot"].to(device, non_blocking=True)
            aaindex = batch["aaindex"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            esm_len = batch["esm_len"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            payload = model.compute_outputs(esm, one_hot, aaindex, esm_len=esm_len)
            logits = payload["logits"]
            loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            running_loss += float(loss.item())

        epoch_loss = running_loss / max(len(train_loader), 1)
        logs["train_loss"].append(epoch_loss)
        logs["lr"].append(float(optimizer.param_groups[0]["lr"]))
        logs["stage_name"].append(stage_name)

        with torch.no_grad():
            model.eval()
            val_running_loss = 0.0
            for batch in val_loader:
                esm = batch["esm"].to(device)
                one_hot = batch["one_hot"].to(device)
                aaindex = batch["aaindex"].to(device)
                labels = batch["label"].to(device)
                esm_len = batch["esm_len"].to(device)
                payload = model.compute_outputs(esm, one_hot, aaindex, esm_len=esm_len)
                logits = payload["logits"]
                val_loss = criterion(logits, labels)
                val_running_loss += float(val_loss.item())
            current_val_loss = val_running_loss / max(len(val_loader), 1)

        val_metrics = evaluate_model(model, val_loader, device, fixed_threshold=fixed_threshold)
        current_val_f1 = float(val_metrics["f1"])
        current_val_mcc = float(val_metrics["mcc"])
        logs["val_loss"].append(current_val_loss)
        logs["val_f1"].append(current_val_f1)
        logs["val_mcc"].append(current_val_mcc)

        if config.checkpoint_selection_metric == "validation_loss":
            improved = current_val_loss < best_val_loss - config.min_delta
        else:
            improved = current_val_f1 > best_f1 + config.min_delta

        if improved:
            best_val_loss = current_val_loss
            best_f1 = current_val_f1
            best_val_mcc = current_val_mcc
            best_epoch = epoch + 1
            best_stage = stage_name
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_since_improve = 0
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_since_improve += 1

        if verbose and ((epoch + 1) % 5 == 0 or epoch == config.epochs - 1):
            print(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"stage={stage_name} "
                f"train_loss={epoch_loss:.4f} "
                f"val_loss={current_val_loss:.4f} "
                f"val_f1={current_val_f1:.4f} "
                f"val_mcc={current_val_mcc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
        if config.patience > 0 and epochs_since_improve >= config.patience:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    if verbose:
        if config.checkpoint_selection_metric == "validation_loss":
            print(
                f"Training finished. best_epoch={best_epoch} best_stage={best_stage} "
                f"best_val_loss={best_val_loss:.4f} "
                f"val_f1_at_best_loss={best_f1:.4f} "
                f"val_mcc_at_best_loss={best_val_mcc:.4f}"
            )
        else:
            print(
                f"Training finished. best_epoch={best_epoch} best_stage={best_stage} "
                f"best_val_f1={best_f1:.4f} "
                f"val_loss_at_best_f1={best_val_loss:.4f}"
            )
    logs["best_epoch"] = int(best_epoch)
    logs["best_stage"] = best_stage
    logs["best_validation_f1"] = float(best_f1)
    logs["best_validation_loss"] = float(best_val_loss)
    logs["best_validation_mcc"] = float(best_val_mcc)
    logs["best_validation_loss_at_best_f1"] = float(best_val_loss)
    logs["best_validation_mcc_at_best_f1"] = float(best_val_mcc)
    logs["best_validation_f1_at_best_loss"] = float(best_f1)
    logs["best_validation_mcc_at_best_loss"] = float(best_val_mcc)
    logs["checkpoint_selection_metric"] = str(config.checkpoint_selection_metric)
    logs["effective_training_schedule"] = "single_stage"
    logs["pos_weight"] = float(adjusted_pos_weight)
    logs["pos_weight_cap"] = float(config.pos_weight_cap)
    return best_f1, logs


def run_cnn_rescue_train_val_test(
    train_dataset,
    val_dataset,
    test_dataset,
    config: CNNRescueTrainingConfig,
    device: str | torch.device,
    *,
    collate_fn=None,
    checkpoint_path: Optional[str | Path] = None,
    training_curve_path: Optional[str | Path] = None,
    evaluation_artifacts_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    start_time = time.time()
    if config.random_seed is not None:
        set_global_seed(config.random_seed, deterministic=config.deterministic)
    model = build_cnn_rescue_model(config, device)
    gate_bias_init_prob = model.get_cnn_gate_bias_probability() if hasattr(model, "get_cnn_gate_bias_probability") else None

    train_loader = create_dataloader(train_dataset, config, shuffle=True, collate_fn=collate_fn, seed=config.random_seed)
    val_loader = create_dataloader(
        val_dataset,
        config,
        shuffle=False,
        collate_fn=collate_fn,
        seed=None if config.random_seed is None else config.random_seed + 1,
    )
    best_f1, logs = train_cnn_rescue_model(
        model,
        train_loader,
        val_loader,
        device,
        config,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
    )
    if config.threshold_policy == "fixed_prob":
        val_threshold = probability_to_logit(float(config.fixed_threshold_prob))
    else:
        val_metrics = evaluate_model(model, val_loader, device)
        val_threshold = float(val_metrics["threshold_logit"])

    if training_curve_path is not None:
        plot_training_curves(logs, training_curve_path, "CNN Rescue Parallel Training Curves")

    val_metrics = evaluate_model(
        model,
        val_loader,
        device,
        fixed_threshold=val_threshold,
        return_raw=True,
        collect_diagnostics=True,
    )
    test_loader = create_dataloader(test_dataset, config, shuffle=False, collate_fn=collate_fn)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        fixed_threshold=val_threshold,
        return_raw=True,
        collect_diagnostics=True,
    )
    if config.threshold_policy == "fixed_prob":
        test_metrics["threshold_source"] = f"fixed_{float(config.fixed_threshold_prob):.1f}"
    else:
        test_metrics["threshold_source"] = "validation_max_f1_then_fixed_on_test"
    gate_bias_final_prob = model.get_cnn_gate_bias_probability() if hasattr(model, "get_cnn_gate_bias_probability") else None

    if evaluation_artifacts_path is not None:
        evaluation_artifacts_path = Path(evaluation_artifacts_path)
        evaluation_artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            evaluation_artifacts_path,
            val_logits=val_metrics["logits_array"],
            val_labels=val_metrics["labels_array"],
            test_logits=test_metrics["logits_array"],
            test_labels=test_metrics["labels_array"],
        )

    return {
        "model": model,
        "best_validation_f1": float(best_f1),
        "validation_threshold_logit": float(val_threshold),
        "validation_threshold_prob": float(logit_to_probability(val_threshold)),
        "threshold_policy": config.threshold_policy,
        "training_logs": logs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "duration_seconds": float(time.time() - start_time),
        "cnn_gate_bias_init_prob": gate_bias_init_prob,
        "cnn_gate_bias_final_prob": gate_bias_final_prob,
    }


def _default_cnn_rescue_config(
    esm_dim: int,
    train_seed: int,
    overrides: Optional[Dict[str, Any]] = None,
) -> CNNRescueTrainingConfig:
    base = {
        "esm_dim": esm_dim,
        "random_seed": train_seed,
        "deterministic": False,
        "use_onehot": False,
        "use_aaindex": False,
    }
    if overrides:
        base.update(overrides)
    return CNNRescueTrainingConfig(**base)


def run_cnn_rescue_main_model_from_feature_path(
    *,
    feature_path: str | Path,
    output_dir: str | Path,
    split_seed: int,
    train_seed: int,
    device: str,
    manifest: Dict[str, Any],
    training_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    feature_records = load_feature_records(feature_path)
    manifest_path = output_dir / "split_manifest.json"
    write_json(manifest, manifest_path)
    train_ds, val_ds, test_ds, meta = build_datasets_from_manifest(feature_records, manifest)
    config = _default_cnn_rescue_config(meta["esm_dim"], train_seed, overrides=training_overrides)
    training_result = run_cnn_rescue_train_val_test(
        train_ds,
        val_ds,
        test_ds,
        config,
        device,
        collate_fn=window_collate_fn,
        checkpoint_path=output_dir / "best_model.pt",
        training_curve_path=output_dir / "training_curves.png",
        evaluation_artifacts_path=output_dir / "evaluation_artifacts.npz",
        verbose=verbose,
    )

    split_payload = {
        "split_seed": split_seed,
        "train_seed": train_seed,
        "feature_path": str(Path(feature_path).resolve()),
        "ids_train": list(manifest["ids_train"]),
        "ids_val": list(manifest["ids_val"]),
        "ids_test": list(manifest["ids_test"]),
        "train": {**dataset_stats(train_ds), "records": dataset_records(train_ds)},
        "val": {**dataset_stats(val_ds), "records": dataset_records(val_ds)},
        "test": {**dataset_stats(test_ds), "records": dataset_records(test_ds)},
    }
    write_json(split_payload, output_dir / "split_manifest.json")
    test_metrics = dict(training_result["test_metrics"])
    val_metrics = dict(training_result["val_metrics"])
    test_metrics.pop("logits_array", None)
    test_metrics.pop("labels_array", None)
    val_metrics.pop("logits_array", None)
    val_metrics.pop("labels_array", None)
    write_json(test_metrics, output_dir / "metrics.json")
    write_json(val_metrics, output_dir / "validation_metrics.json")
    write_json(
        {
            "split_seed": split_seed,
            "train_seed": train_seed,
            "duration_seconds": training_result["duration_seconds"],
            "best_validation_f1": training_result["best_validation_f1"],
            "best_epoch": int(training_result["training_logs"].get("best_epoch", 0)),
            "best_stage": training_result["training_logs"].get("best_stage", ""),
            "best_validation_loss": float(
                training_result["training_logs"].get("best_validation_loss", float("nan"))
            ),
            "best_validation_mcc": float(
                training_result["training_logs"].get("best_validation_mcc", float("nan"))
            ),
            "best_validation_loss_at_best_f1": float(
                training_result["training_logs"].get("best_validation_loss_at_best_f1", float("nan"))
            ),
            "best_validation_mcc_at_best_f1": float(
                training_result["training_logs"].get("best_validation_mcc_at_best_f1", float("nan"))
            ),
            "best_validation_f1_at_best_loss": float(
                training_result["training_logs"].get("best_validation_f1_at_best_loss", float("nan"))
            ),
            "best_validation_mcc_at_best_loss": float(
                training_result["training_logs"].get("best_validation_mcc_at_best_loss", float("nan"))
            ),
            "checkpoint_selection_metric": str(
                training_result["training_logs"].get("checkpoint_selection_metric", config.checkpoint_selection_metric)
            ),
            "effective_training_schedule": str(
                training_result["training_logs"].get("effective_training_schedule", "single_stage")
            ),
            "validation_threshold_logit": training_result["validation_threshold_logit"],
            "validation_threshold_prob": training_result["validation_threshold_prob"],
            "threshold_policy": training_result.get("threshold_policy", "validation_f1"),
            "threshold_source": training_result["test_metrics"].get("threshold_source", "unknown"),
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "pos_weight": float(training_result["training_logs"].get("pos_weight", float("nan"))),
            "pos_weight_cap": float(config.pos_weight_cap),
            "model_family": config.model_family,
            "battlefield": config.battlefield,
            "evaluation_protocols": list(config.evaluation_protocols),
            "cnn_branch_mode": config.cnn_branch_mode,
            "cnn_pooling_mode": config.cnn_pooling_mode,
            "main_pooling_mode": config.main_pooling_mode,
            "cnn_only_mode": bool(config.cnn_only_mode),
            "cnn_branch_gate_mode": config.cnn_branch_gate_mode,
            "training_schedule": dict(config.training_schedule),
            "checkpoint_selection_metric": str(config.checkpoint_selection_metric),
            "top_k": int(config.top_k),
            "cnn_scalar_gate_init_prob": float(config.cnn_scalar_gate_init_prob),
            "cnn_gate_bias_init_prob": training_result.get("cnn_gate_bias_init_prob"),
            "cnn_gate_bias_final_prob": training_result.get("cnn_gate_bias_final_prob"),
        },
        output_dir / "training_meta.json",
    )

    metrics = training_result["test_metrics"]
    row = {
        "status": "success",
        "split_seed": int(split_seed),
        "train_seed": int(train_seed),
        "feature_path": str(Path(feature_path).resolve()),
        "run_dir": str(output_dir.resolve()),
        "checkpoint_path": str((output_dir / "best_model.pt").resolve()),
        "training_curve_path": str((output_dir / "training_curves.png").resolve()),
        "metrics_path": str((output_dir / "metrics.json").resolve()),
        "validation_metrics_path": str((output_dir / "validation_metrics.json").resolve()),
        "split_manifest_path": str(Path(manifest_path).resolve()),
        "training_meta_path": str((output_dir / "training_meta.json").resolve()),
        "evaluation_artifacts_path": str((output_dir / "evaluation_artifacts.npz").resolve()),
        "best_validation_f1": float(training_result["best_validation_f1"]),
        "best_epoch": int(training_result["training_logs"].get("best_epoch", 0)),
        "best_stage": training_result["training_logs"].get("best_stage", ""),
        "best_validation_loss": float(
            training_result["training_logs"].get("best_validation_loss", float("nan"))
        ),
        "best_validation_mcc": float(
            training_result["training_logs"].get("best_validation_mcc", float("nan"))
        ),
        "best_validation_loss_at_best_f1": float(
            training_result["training_logs"].get("best_validation_loss_at_best_f1", float("nan"))
        ),
        "best_validation_mcc_at_best_f1": float(
            training_result["training_logs"].get("best_validation_mcc_at_best_f1", float("nan"))
        ),
        "best_validation_f1_at_best_loss": float(
            training_result["training_logs"].get("best_validation_f1_at_best_loss", float("nan"))
        ),
        "best_validation_mcc_at_best_loss": float(
            training_result["training_logs"].get("best_validation_mcc_at_best_loss", float("nan"))
        ),
        "checkpoint_selection_metric": str(
            training_result["training_logs"].get("checkpoint_selection_metric", config.checkpoint_selection_metric)
        ),
        "effective_training_schedule": str(
            training_result["training_logs"].get("effective_training_schedule", "single_stage")
        ),
        "validation_threshold_logit": float(training_result["validation_threshold_logit"]),
        "validation_threshold_prob": float(training_result["validation_threshold_prob"]),
        "threshold_policy": str(training_result.get("threshold_policy", "validation_f1")),
        "duration_seconds": float(training_result["duration_seconds"]),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "test_size": int(len(test_ds)),
        "train_pos": int(dataset_stats(train_ds)["positive_count"]),
        "val_pos": int(dataset_stats(val_ds)["positive_count"]),
        "test_pos": int(dataset_stats(test_ds)["positive_count"]),
        "train_neg": int(dataset_stats(train_ds)["negative_count"]),
        "val_neg": int(dataset_stats(val_ds)["negative_count"]),
        "test_neg": int(dataset_stats(test_ds)["negative_count"]),
        "weight_decay": float(config.weight_decay),
        "label_smoothing": float(config.label_smoothing),
        "pos_weight": float(training_result["training_logs"].get("pos_weight", float("nan"))),
        "pos_weight_cap": float(config.pos_weight_cap),
        "model_family": str(config.model_family),
        "battlefield": str(config.battlefield),
        "cnn_branch_mode": str(config.cnn_branch_mode),
        "cnn_pooling_mode": str(config.cnn_pooling_mode),
        "cnn_branch_gate_mode": str(config.cnn_branch_gate_mode),
        "main_pooling_mode": str(config.main_pooling_mode),
        "cnn_only_mode": bool(config.cnn_only_mode),
        "training_schedule": str(config.training_schedule),
        "checkpoint_selection_metric_config": str(config.checkpoint_selection_metric),
        "cnn_scalar_gate_init_prob": float(config.cnn_scalar_gate_init_prob),
        "cnn_gate_mean": metrics.get("cnn_gate_mean"),
        "cnn_gate_std": metrics.get("cnn_gate_std"),
        "branch_gate_vector": metrics.get("branch_gate_vector"),
        "topk_motif_scores_summary": metrics.get("topk_motif_scores_summary"),
        "topk_positions_summary": metrics.get("topk_positions_summary"),
        "cnn_gate_bias_init_prob": training_result.get("cnn_gate_bias_init_prob"),
        "cnn_gate_bias_final_prob": training_result.get("cnn_gate_bias_final_prob"),
    }
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "mcc",
        "specificity",
        "threshold_prob",
        "threshold_logit",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        if key in metrics:
            row[key] = float(metrics[key]) if key not in {"tn", "fp", "fn", "tp"} else int(metrics[key])
    row["threshold_source"] = str(metrics.get("threshold_source", "unknown"))
    row["confusion_matrix"] = str(metrics.get("confusion_matrix"))
    return row
