import importlib, unittest, tempfile, json, hashlib, zipfile
from pathlib import Path


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.artifacts")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Artifact interface missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        (self.p / "source").mkdir()
        (self.p / "source/a.bin").write_bytes(b"abc")
        self.manifest = {
            "assets": {
                "a": {
                    "filename": "a.bin",
                    "bytes": 3,
                    "sha256": hashlib.sha256(b"abc").hexdigest(),
                    "url": None,
                }
            }
        }

    def test_offline_verified_copy_and_cache(self):
        p = self.api.fetch_asset(
            "a", self.manifest, self.p / "cache", offline_dir=self.p / "source"
        )
        self.assertEqual(p.read_bytes(), b"abc")
        self.assertEqual(self.api.fetch_asset("a", self.manifest, self.p / "cache"), p)

    def test_bad_hash_and_existing_content_not_overwritten(self):
        (self.p / "source/a.bin").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            self.api.fetch_asset(
                "a", self.manifest, self.p / "cache", offline_dir=self.p / "source"
            )
        self.assertFalse((self.p / "cache/a.bin").exists())

    def test_zip_traversal_rejected_before_extract(self):
        p = self.p / "bad.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("../escape", "x")
        with self.assertRaises(ValueError):
            self.api.extract_zip(p, self.p / "out")
        self.assertFalse((self.p / "out").exists())
        self.assertFalse((self.p / "escape").exists())

    def test_unpublished_url_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "published"):
            self.api.fetch_asset("a", self.manifest, self.p / "cache")

    def test_registered_main_rejects_changed_attention_heads(self):
        from hiantitoxinpred.model import ModelConfig

        root = self.p / "models"
        root.mkdir()
        (root / "best_model.pt").write_bytes(b"abc")
        item = dict(
            seed=42,
            filename="best_model.pt",
            bytes=3,
            sha256=hashlib.sha256(b"abc").hexdigest(),
        )
        (root / "MODEL_INDEX.json").write_text(
            json.dumps(dict(best_seed=42, models=[item]))
        )
        config = ModelConfig().to_dict()
        config["heads"] = 16
        (root / "model_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "registered Main"):
            self.api.load_registered_model(root)
