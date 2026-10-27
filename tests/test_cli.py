import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class CLITests(unittest.TestCase):
    def setUp(self):
        try:
            self.cli = importlib.import_module("hiantitoxinpred.cli")
        except ModuleNotFoundError:
            self.cli = None
        self.assertIsNotNone(self.cli, "CLI not implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fasta = self.root / "input.fa"
        self.fasta.write_text(">short\nACDE\n>long\n" + "A" * 205 + "\n")

    def test_prediction_output_and_no_overwrite(self):
        class Encoder:
            identity = {"test": True}

            def __init__(self, *a, **k):
                pass

            def encode(self, seqs, **k):
                return [np.zeros((len(s), 1024), np.float32) for s in seqs]

        output = self.root / "prediction.csv"
        with (
            patch.object(self.cli, "ProtT5Encoder", Encoder),
            patch.object(
                self.cli, "load_registered_model", return_value=(object(), {"seed": 42})
            ),
            patch.object(
                self.cli, "predict_features", return_value=np.array([-1, 2], np.float32)
            ),
        ):
            args = [
                "predict",
                "--fasta",
                str(self.fasta),
                "--models",
                "none",
                "--encoder",
                "none",
                "--output",
                str(output),
            ]
            self.cli.main(args)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["id"] for r in rows], ["short", "long"])
            self.assertEqual(rows[1]["input_length"], "200")
            self.assertEqual(rows[1]["truncated"], "True")
            self.assertEqual([r["prediction"] for r in rows], ["0", "1"])
            with self.assertRaises(FileExistsError):
                self.cli.main(args)

    def test_configuration_rejects_extra_and_wrong_protocol(self):
        p = self.root / "config.json"
        p.write_text(json.dumps({"training": {}, "model": {}, "unexpected": 1}))
        with self.assertRaises(ValueError):
            self.cli.read_config(p)

    def test_test_evaluation_rejects_unlocked_run(self):
        run = self.root / "run"
        run.mkdir()
        (run / "L.pt").write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "lock"):
            self.cli.load_run_model(run / "L.pt", "cpu")

    def test_evaluation_rejects_altered_model_config(self):
        import hashlib
        from hiantitoxinpred.model import ModelConfig

        run = self.root / "run"
        run.mkdir()
        (run / "L.pt").write_bytes(b"abc")
        (run / "SELECTION_LOCK.json").write_text(
            json.dumps(dict(weights_sha256=hashlib.sha256(b"abc").hexdigest()))
        )
        for name in ("COMPLETE.json", "RELOAD_VERIFICATION.json"):
            (run / name).write_text(json.dumps(dict(status="PASS")))
        config = ModelConfig().to_dict()
        (run / "CONFIG.json").write_text(json.dumps(dict(model=config, seed=42)))
        config["heads"] = 16
        (run / "model_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.cli.load_run_model(run / "L.pt", "cpu")

    def test_split_model_archives_install_without_manifest_collision(self):
        import zipfile, hashlib

        source = self.root / "source"
        source.mkdir()
        assets = {}
        for name in ("metadata", "weights"):
            p = source / (name + ".zip")
            with zipfile.ZipFile(p, "w") as z:
                z.writestr(name + ".bin", name)
                z.writestr("MEMBER_CHECKSUMS.json", json.dumps({"name": name}))
            assets[name] = dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                url=None,
                destination="models",
                archive=True,
                member_manifest=name + "_CHECKSUMS.json",
            )
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(dict(assets=assets, groups={"all": ["metadata", "weights"]}))
        )
        dest = self.root / "installed"
        self.cli.main(
            [
                "download",
                "--assets",
                "all",
                "--manifest",
                str(manifest),
                "--offline-dir",
                str(source),
                "--directory",
                str(dest),
            ]
        )
        self.assertTrue((dest / "models/metadata.bin").exists())
        self.assertTrue((dest / "models/weights.bin").exists())
