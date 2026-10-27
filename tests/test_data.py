import importlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch


class DataTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.data")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Input data API is not implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fasta(self, text):
        p = self.root / "input.fa"
        p.write_text(text)
        return self.api.read_fasta(p)

    def test_fasta_preserves_order_and_normalized_input_identity(self):
        rows = self.fasta(">b description\nac d\nuzob\n>a\n" + "M" * 201 + "\n")
        self.assertEqual([r.id for r in rows], ["b", "a"])
        self.assertEqual(rows[0].sequence, "ACDUZOB")
        self.assertEqual(rows[0].encoder_sequence, "ACDXXXX")
        self.assertEqual(
            rows[0].sequence_sha256, hashlib.sha256(b"ACDUZOB").hexdigest()
        )
        self.assertEqual(rows[1].original_length, 201)
        self.assertEqual(rows[1].input_length, 200)
        self.assertTrue(rows[1].truncated)

    def test_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.fasta(">a first\nAC\n>a second\nDE\n")

    def test_reject_empty_or_invalid_fasta(self):
        for text in (
            "",
            ">a\n",
            "AC\n>a\nAC",
            ">a\nAC*",
            ">a\nAC-",
            ">\nAC",
            ">a\nAJC",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.fasta(text)

    def test_reject_non_ascii_residues_before_case_normalization(self):
        # Unicode uppercasing must not manufacture extra amino-acid residues.
        with self.assertRaises(ValueError):
            self.fasta(">a\nAßC")

    def test_collation_preserves_features_and_mask(self):
        values = [torch.ones(3, 4), torch.full((2, 4), 2.0)]
        x, mask = self.api.collate_features(values)
        self.assertEqual(tuple(x.shape), (2, 3, 4))
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(x[1, 2].sum().item(), 0)
        torch.testing.assert_close(x[1, :2], values[1])

    def test_one_residue_and_explicit_width(self):
        x, mask = self.api.collate_features([torch.ones(1, 4)])
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(mask.sum().item(), 1)
        x, mask = self.api.collate_features([torch.ones(19, 4)], width=200)
        self.assertEqual(x.shape[1], 200)
        self.assertEqual(mask.sum().item(), 19)

    def test_collation_rejects_invalid_input(self):
        for values in (
            [],
            [torch.ones(0, 4)],
            [torch.ones(201, 4)],
            [torch.ones(2, 4).double()],
            [torch.full((2, 4), float("nan"))],
            [torch.ones(2, 4), torch.ones(2, 3)],
        ):
            with self.subTest(values=str(values)[:40]), self.assertRaises(ValueError):
                self.api.collate_features(values)
        with self.assertRaises(ValueError):
            self.api.collate_features([torch.ones(20, 4)], width=19)

    def write_main(self, manifest=None, sequence="AC"):
        import csv

        rows = [
            ("a", sequence, "1", "train"),
            ("b", "DE", "0", "val"),
            ("c", "FG", "1", "test"),
        ]
        p = self.root / "main.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label", "split", "sequence_sha256"])
            for id_, seq, y, split in rows:
                w.writerow(
                    [id_, seq, y, split, hashlib.sha256(seq.encode()).hexdigest()]
                )
        q = self.root / "splits.json"
        q.write_text(
            json.dumps(
                manifest or {"ids_train": ["a"], "ids_val": ["b"], "ids_test": ["c"]}
            )
        )
        return p, q

    def test_registered_members_preserve_manifest_order(self):
        data = self.api.read_main(*self.write_main())
        self.assertEqual([r["id"] for r in data["train"]], ["a"])
        self.assertEqual(data["val"][0]["label"], 0)

    def test_registered_members_reject_overlap_or_missing(self):
        for m in (
            {"ids_train": ["a"], "ids_val": ["a"], "ids_test": ["c"]},
            {"ids_train": ["a"], "ids_val": ["b"], "ids_test": ["missing"]},
        ):
            with self.assertRaises(ValueError):
                self.api.read_main(*self.write_main(m))

    def test_registered_sequences_not_silently_normalized(self):
        with self.assertRaises(ValueError):
            self.api.read_main(*self.write_main(sequence="ac"))

    def test_registered_data_identity_rejects_consistent_label_edit(self):
        import shutil

        self.assertTrue(hasattr(self.api, "read_registered_main"))
        csv_path, split_path = self.write_main()
        for source, name in [
            (csv_path, "processed_sequences.csv"),
            (split_path, "split_manifest.json"),
        ]:
            shutil.copyfile(source, self.root / name)
        hashes = {
            n: hashlib.sha256((self.root / n).read_bytes()).hexdigest()
            for n in ("processed_sequences.csv", "split_manifest.json")
        }
        feature_files = {"index.json": "abc", "per_residue.npy": "def"}
        (self.root / "DATA_IDENTITY.json").write_text(
            json.dumps(dict(files=hashes, feature_files=feature_files))
        )
        rows, expected = self.api.read_registered_main(self.root)
        self.assertEqual(rows["train"][0]["label"], 1)
        self.assertEqual(expected, feature_files)
        p = self.root / "processed_sequences.csv"
        p.write_text(p.read_text().replace("a,AC,1,train", "a,AC,0,train"))
        # Internal CSV/manifest consistency still passes, but registered identity must not.
        self.assertEqual(
            self.api.read_main(p, self.root / "split_manifest.json")["train"][0][
                "label"
            ],
            0,
        )
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.api.read_registered_main(self.root)
