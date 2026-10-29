import csv
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hiantitoxinpred import cli
from hiantitoxinpred.checkpoints import file_sha256
from hiantitoxinpred.encoder import REGISTERED_CHECKSUMS


class SmallEncoder:
    """Offline stand-in for expensive encoding; storage and validation stay real."""

    identity = dict(
        name="ProtT5-XL-U50-half-encoder",
        files=REGISTERED_CHECKSUMS,
        precision="cpu_fp32",
        preprocessing="cap200-before-UZOB-to-X; spaced tokens; exclude special/padding; float32 storage",
        version=1,
    )

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, sequences, *, batch_size=2):
        return [np.full((len(s), 1024), ord(s[0]) / 100, np.float32) for s in sequences]


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.output = self.root / "features"
        # CSV order deliberately differs from the fixed split order.
        rows = [
            ("c", "UZOB", 1, "test"),
            ("a", "AC", 1, "train"),
            ("b", "DEF", 0, "val"),
        ]
        with (self.data / "processed_sequences.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label", "split", "sequence_sha256"])
            for id_, sequence, label, split in rows:
                w.writerow(
                    [
                        id_,
                        sequence,
                        label,
                        split,
                        hashlib.sha256(sequence.encode()).hexdigest(),
                    ]
                )
        (self.data / "split_manifest.json").write_text(
            json.dumps(dict(ids_train=["a"], ids_val=["b"], ids_test=["c"]))
        )
        self.files = {
            n: file_sha256(self.data / n)
            for n in ("processed_sequences.csv", "split_manifest.json")
        }
        (self.data / "DATA_IDENTITY.json").write_text(
            json.dumps(
                dict(
                    files=self.files,
                    feature_files={
                        "index.json": "old-index",
                        "per_residue.npy": "old-array",
                    },
                )
            )
        )

    def api(self):
        spec = importlib.util.find_spec("hiantitoxinpred.feature_extraction")
        self.assertIsNotNone(spec, "Main feature extraction is not implemented")
        return importlib.import_module("hiantitoxinpred.feature_extraction")

    def extract(self, encoder=None):
        return self.api().extract_main_features(
            self.data, self.output, encoder or SmallEncoder(), batch_size=2
        )

    def refresh_hashes(self):
        hashes = {
            n: file_sha256(self.output / n) for n in ("index.json", "per_residue.npy")
        }
        (self.output / "FEATURES.json").write_text(json.dumps(hashes))
        receipt = json.loads((self.output / "EXTRACTION.json").read_text())
        receipt["feature_files"] = hashes
        (self.output / "EXTRACTION.json").write_text(json.dumps(receipt))

    def load(self):
        from hiantitoxinpred.features import load_main_features

        return load_main_features(self.output, self.data)

    def test_extract_load_fixed_order_and_original_sequence_identity(self):
        self.extract()
        index = json.loads((self.output / "index.json").read_text())
        self.assertEqual([r["id"] for r in index["records"]], ["a", "b", "c"])
        self.assertEqual(
            [(r["start"], r["end"]) for r in index["records"]], [(0, 2), (2, 5), (5, 9)]
        )
        self.assertEqual(
            index["records"][-1]["sequence_sha256"], hashlib.sha256(b"UZOB").hexdigest()
        )
        rows, store = self.load()
        self.assertEqual(rows["test"][0]["sequence"], "UZOB")
        self.assertEqual(store.array.shape, (9, 1024))
        self.assertEqual(store.array.dtype, np.float32)
        np.testing.assert_array_equal(
            store.get("a").numpy(), SmallEncoder().encode(["AC"])[0]
        )

    def test_refuses_overwrite(self):
        self.extract()
        before = file_sha256(self.output / "per_residue.npy")
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(file_sha256(self.output / "per_residue.npy"), before)

    def test_failed_extraction_is_not_loadable(self):
        class Broken(SmallEncoder):
            def encode(self, sequences, *, batch_size=2):
                raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.extract(Broken())
        self.assertFalse((self.output / "EXTRACTION.json").exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load()

    def test_changed_array_rejected(self):
        self.extract()
        a = np.load(self.output / "per_residue.npy", mmap_mode="r+")
        a[0, 0] += 1
        a.flush()
        del a
        with self.assertRaisesRegex(ValueError, "SHA256|hash"):
            self.load()

    def test_changed_encoder_rejected_even_with_new_hash(self):
        self.extract()
        path = self.output / "ENCODER_IDENTITY.json"
        identity = json.loads(path.read_text())
        identity["files"]["config.json"] = "different"
        path.write_text(json.dumps(identity))
        receipt = json.loads((self.output / "EXTRACTION.json").read_text())
        receipt["encoder_identity_sha256"] = file_sha256(path)
        (self.output / "EXTRACTION.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "[Ee]ncoder"):
            self.load()

    def test_wrong_label_split_or_offsets_rejected_with_rehashed_index(self):
        self.extract()
        path = self.output / "index.json"
        original = path.read_text()
        for change in (dict(label=0), dict(split="test"), dict(start=1, end=3)):
            index = json.loads(original)
            index["records"][0].update(change)
            path.write_text(json.dumps(index))
            self.refresh_hashes()
            with self.assertRaises(ValueError):
                self.load()
        path.write_text(original)
        self.refresh_hashes()
        self.load()

    def test_nonfinite_encoder_output_rejected(self):
        class Broken(SmallEncoder):
            def encode(self, sequences, *, batch_size=2):
                return [np.full((len(s), 1024), np.nan, np.float32) for s in sequences]

        with self.assertRaises(ValueError):
            self.extract(Broken())
        self.assertFalse((self.output / "EXTRACTION.json").exists())

    def test_different_data_provenance_rejected(self):
        self.extract()
        path = self.output / "EXTRACTION.json"
        r = json.loads(path.read_text())
        r["data_files"]["processed_sequences.csv"] = "different"
        path.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError, "[Dd]ata|[Ii]nput"):
            self.load()

    def test_cli_extract_and_train_use_generated_features(self):
        self.api()
        with patch.object(cli, "ProtT5Encoder", SmallEncoder):
            cli.main(
                [
                    "extract-features",
                    "--data-dir",
                    str(self.data),
                    "--encoder",
                    "unused",
                    "--output",
                    str(self.output),
                ]
            )
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps(dict(training={}, model={})))
        with patch.object(cli, "fit") as fit:
            cli.main(
                [
                    "train",
                    "--data-dir",
                    str(self.data),
                    "--features-dir",
                    str(self.output),
                    "--config",
                    str(cfg),
                    "--output",
                    str(self.root / "run"),
                ]
            )
        self.assertEqual(fit.call_args.args[0].ids, ["a"])
        self.assertEqual(fit.call_args.args[1].ids, ["b"])

    def test_legacy_original_features_still_load(self):
        self.extract()
        identity = json.loads((self.data / "DATA_IDENTITY.json").read_text())
        identity["feature_files"] = json.loads(
            (self.output / "FEATURES.json").read_text()
        )
        (self.data / "DATA_IDENTITY.json").write_text(json.dumps(identity))
        (self.output / "EXTRACTION.json").unlink()
        (self.output / "FEATURES.json").unlink()
        self.load()
