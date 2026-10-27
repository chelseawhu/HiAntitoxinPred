"""C1 trainer: train/validation only, true-minimum-loss L checkpoint."""

from dataclasses import dataclass, asdict
import csv, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import HiAntitoxinPred, ModelConfig
from .data import collate_features
from .metrics import binary_metrics
from .selection import LossSelection
from .checkpoints import file_sha256


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temp, path)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-6
    weight_decay: float = 0.05
    batch_size: int = 16
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-5

    def __post_init__(self):
        for k in ("batch_size", "max_epochs", "patience"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(k)
        if self.batch_size < 2:
            raise ValueError("BatchNorm training requires batch_size>=2")
        for k in ("learning_rate", "weight_decay", "min_delta"):
            v = getattr(self, k)
            if not math.isfinite(v) or v < 0:
                raise ValueError(k)
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


def batch(samples):
    ids, x, y = zip(*samples)
    features, mask = collate_features(x)
    return list(ids), features, mask, torch.tensor(y, dtype=torch.float32)


def predict_dataset(model, dataset, batch_size=16, device="cpu", criterion=None):
    model.eval()
    zs = []
    ys = []
    ids = []
    loss_sum = 0.0
    with torch.inference_mode():
        for names, x, mask, y in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=batch
        ):
            z = model(x.to(device), mask.to(device))
            target = y.to(device)
            if not torch.isfinite(z).all():
                raise ValueError("Non-finite evaluation logits")
            if criterion is not None:
                loss_sum += float(criterion(z, target)) * len(y)
            ids.extend(names)
            zs.extend(z.cpu().tolist())
            ys.extend(y.tolist())
    if not ids:
        raise ValueError("Empty evaluation dataset")
    return (
        ids,
        np.asarray(ys, np.int64),
        np.asarray(zs, np.float32),
        loss_sum / len(ids),
    )


def save_predictions(path, ids, y, z):
    with Path(path).open("x", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "label", "raw_logit", "probability", "prediction"])
        probability = torch.from_numpy(z).sigmoid().numpy()
        w.writerows(
            zip(
                ids,
                y.tolist(),
                z.tolist(),
                probability.tolist(),
                (z >= 0).astype(int).tolist(),
            )
        )


def fit(train, val, output, config=None, model_config=None, *, seed=42, device="cpu"):
    config = config or TrainingConfig()
    model_config = model_config or ModelConfig()
    if train.split != "train" or val.split != "val":
        raise ValueError("fit accepts only train and val datasets")
    if set(train.ids) & set(val.ids):
        raise ValueError("Train/validation ID overlap")
    if len(set(train.ids)) != len(train.ids) or len(set(val.ids)) != len(val.ids):
        raise ValueError("Duplicate input ID")
    if len(train) < config.batch_size or not len(val):
        raise ValueError("Insufficient data")
    if set(train.labels) != {0, 1} or set(val.labels) != {0, 1}:
        raise ValueError("Both binary classes required")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "validation").mkdir()
    start = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    model = HiAntitoxinPred(model_config).to(device)
    pos = sum(train.labels)
    weight = math.sqrt((len(train) - pos) / pos)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-7,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=batch,
        generator=generator,
    )
    selection = LossSelection(config.patience, config.min_delta)
    history = []
    write_json(out / "model_config.json", model_config.to_dict())
    write_json(
        out / "CONFIG.json",
        dict(
            training=asdict(config),
            model=model_config.to_dict(),
            seed=seed,
            loss="BCEWithLogitsLoss",
            pos_weight=weight,
            sampling="natural",
            label_smoothing=0,
            train_feature_clip=[-10, 10],
            validation_feature_clip=None,
            optimizer=dict(name="AdamW", betas=[0.9, 0.95], eps=1e-7),
            gradient_clip=0.5,
            drop_last_train=True,
            checkpoint="L: true minimum validation objective loss",
            geometry="longest_in_batch",
            initialization="refactored effective network; not bitwise historical RNG trajectory",
        ),
    )
    write_json(
        out / "INPUT_IDENTITY.json",
        dict(
            train=train.identity, val=val.identity, train_ids=train.ids, val_ids=val.ids
        ),
    )
    write_json(
        out / "ENVIRONMENT.json",
        dict(
            python=sys.version,
            torch=str(torch.__version__),
            numpy=np.__version__,
            device=str(device),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            seed=seed,
            deterministic=False,
            threads=torch.get_num_threads(),
        ),
    )
    try:
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            losses = []
            for _, x, mask, y in loader:
                optimizer.zero_grad(set_to_none=True)
                z = model(x.to(device).clamp(-10, 10), mask.to(device))
                loss = criterion(z, y.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if not torch.isfinite(norm):
                    raise ValueError("Non-finite gradient norm")
                optimizer.step()
                losses.append(float(loss.detach()))
            ids, y, z, val_loss = predict_dataset(
                model, val, config.batch_size, device, criterion
            )
            selected = selection.update(epoch, val_loss)
            metrics = binary_metrics(y, z)
            row = dict(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                val_loss=val_loss,
                val_F1=metrics["F1"],
                val_MCC=metrics["MCC"],
                wait=selection.wait,
                reference_loss=selection.reference_loss,
                L_epoch=selection.best_epoch,
            )
            history.append(row)
            np.savez_compressed(
                out / "validation" / f"epoch{epoch:03d}.npz",
                ids=np.asarray(ids),
                labels=y,
                raw_logits=z,
            )
            if selected:
                state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                torch.save(state, out / "L.pt.tmp")
                os.replace(out / "L.pt.tmp", out / "L.pt")
            write_json(out / "history.json", history)
            write_json(
                out / "progress.json",
                dict(status="training", **row, seconds=time.monotonic() - start),
            )
            print(json.dumps(row), flush=True)
            if selection.should_stop:
                break
        digest = file_sha256(out / "L.pt")
        lock = dict(
            L_epoch=selection.best_epoch,
            val_loss=selection.best_loss,
            weights_sha256=digest,
            test_evaluated=False,
            stop_epoch=epoch,
            stop_reason="patience" if selection.should_stop else "max_epochs",
        )
        write_json(out / "SELECTION_LOCK.json", lock)
        model.load_state_dict(
            torch.load(out / "L.pt", map_location=device, weights_only=True),
            strict=True,
        )
        ids, y, z, loss = predict_dataset(
            model, val, config.batch_size, device, criterion
        )
        archived = np.load(
            out / "validation" / f"epoch{selection.best_epoch:03d}.npz",
            allow_pickle=False,
        )
        diff = float(np.max(np.abs(z - archived["raw_logits"])))
        if diff > 1e-4 or not np.array_equal(z >= 0, archived["raw_logits"] >= 0):
            raise ValueError("L reload verification failed")
        save_predictions(out / "L_val.csv", ids, y, z)
        write_json(
            out / "RELOAD_VERIFICATION.json",
            dict(status="PASS", max_logit_difference=diff, classification_changes=0),
        )
        result = dict(
            status="PASS_TRAIN_VAL_NO_TEST",
            history=history,
            pos_weight=weight,
            **lock,
            seconds=time.monotonic() - start,
        )
        write_json(out / "COMPLETE.json", result)
        write_json(out / "progress.json", dict(status="complete", **lock))
        return result
    except BaseException as exc:
        write_json(
            out / "FAILED.json", dict(error=repr(exc), completed_epochs=len(history))
        )
        raise
