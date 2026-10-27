"""Versioned, checksum-verified assets; no unverified checkpoint loading."""

import json, os, shutil, stat, tempfile, urllib.request, zipfile
from pathlib import Path, PurePosixPath
from .checkpoints import file_sha256


def verify(path, entry):
    if (
        Path(path).stat().st_size != entry["bytes"]
        or file_sha256(path) != entry["sha256"]
    ):
        raise ValueError("Asset size/SHA256 mismatch: " + Path(path).name)


def fetch_asset(name, manifest, directory, *, offline_dir=None):
    entry = manifest["assets"][name]
    filename = entry["filename"]
    if Path(filename).name != filename or filename in (".", "..") or "\\" in filename:
        raise ValueError("Unsafe asset filename")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    if target.exists():
        verify(target, entry)
        return target
    if offline_dir is None and not entry.get("url"):
        raise ValueError(
            "Asset URL is not published yet; use --offline-dir with the reviewed local bundle"
        )
    if offline_dir is None and not entry["url"].startswith("https://"):
        raise ValueError("Remote assets require HTTPS")
    temporary = None
    try:
        source = (
            (Path(offline_dir) / filename).open("rb")
            if offline_dir is not None
            else urllib.request.urlopen(entry["url"], timeout=60)
        )
        with source, tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
            temporary = Path(output.name)
            size = 0
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > entry["bytes"]:
                    raise ValueError("Asset exceeds registered size")
                output.write(chunk)
        verify(temporary, entry)
        # Linking refuses to overwrite a concurrently created final file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            verify(target, entry)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_zip(archive, destination, *, max_bytes=20 * 1024**3):
    dest = Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    with zipfile.ZipFile(archive) as z:
        names = set()
        total = 0
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or not path.parts
                or ":" in path.parts[0]
                or str(path) in names
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
            ):
                raise ValueError("Unsafe archive member")
            names.add(str(path))
            total += member.file_size
        if total > max_bytes:
            raise ValueError("Archive exceeds extraction budget")
        dest.mkdir(parents=True)
        for member in z.infolist():
            target = dest.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out)
    return dest


def load_registered_model(directory, *, seed=None, device="cpu"):
    import torch
    from .model import HiAntitoxinPred, ModelConfig

    root = Path(directory)
    index = json.loads((root / "MODEL_INDEX.json").read_text())
    seed = index["best_seed"] if seed is None else seed
    items = [r for r in index["models"] if r["seed"] == seed]
    if len(items) != 1:
        raise ValueError("Unregistered model seed")
    item = items[0]
    p = root / item["filename"]
    if p.resolve().parent != root.resolve():
        raise ValueError("Unsafe model path")
    verify(p, item)
    config = json.loads((root / "model_config.json").read_text())
    if config != ModelConfig().to_dict():
        raise ValueError("Configuration differs from the registered Main architecture")
    model = HiAntitoxinPred(ModelConfig(**config)).to(device)
    model.load_state_dict(
        torch.load(p, map_location=device, weights_only=True), strict=True
    )
    return model.eval(), item
