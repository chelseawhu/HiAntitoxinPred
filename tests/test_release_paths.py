from __future__ import annotations

import unittest
from pathlib import Path


class ReleasePathTest(unittest.TestCase):
    def test_core_files_do_not_contain_local_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checked_roots = [
            root / "src",
            root / "scripts" / "predict",
            root / "scripts" / "train",
            root / "scripts" / "feature_extraction",
            root / "configs",
        ]
        forbidden = ["/" + "data3/", "Bioinformatics" + "_Final", "bib" + "_final_release"]
        offenders: list[str] = []

        for checked_root in checked_roots:
            for path in checked_root.rglob("*"):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for token in forbidden:
                    if token in text:
                        offenders.append(f"{path.relative_to(root)} contains {token}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
