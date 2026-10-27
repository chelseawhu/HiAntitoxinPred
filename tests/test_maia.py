import importlib, unittest


class MaiaTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.maia")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "MAIA missing")

    def test_windows(self):
        self.assertEqual(self.api.full_windows(1), [(1, 1)])
        self.assertEqual(self.api.full_windows(21), [(1, 20), (2, 21)])

    def test_u19_changes_every_window_residue_only(self):
        s = "ACDEFGHIKLMNPQRSTVWYAC"
        a = self.api.mutate_window(s, 1, 20, "U19", 1001, 1)
        self.assertEqual(a, self.api.mutate_window(s, 1, 20, "U19", 1001, 1))
        self.assertEqual(len(a["changes"]), 20)
        self.assertEqual(a["mutant_sequence"][-1], s[-1])
        self.assertNotEqual(
            a["mutant_sequence"],
            self.api.mutate_window(s, 1, 20, "U19", 2001, 1)["mutant_sequence"],
        )

    def test_shuffle_and_ambiguous_refusal(self):
        s = "ACDEFGHIKLMNPQRSTVWYA"
        a = self.api.mutate_window(s, 1, 20, "SHUFFLE_COMPOSITION", 1001, 1)
        self.assertEqual(sorted(a["mutant_sequence"]), sorted(s))
        with self.assertRaises(ValueError):
            self.api.mutate_window("AX", 1, 2, "U19", 1001, 1)

    def test_absolute_after_repeat_mean(self):
        a = self.api.aggregate_signed_sensitivity(
            2, 3.0, [{"start_1based": 1, "end_1based": 2, "mutant_logits": [5.0, 1.0]}]
        )
        self.assertEqual(a["signed_support"], [0.0, 0.0])
        self.assertEqual(a["absolute_sensitivity"], [0.0, 0.0])

    def test_signed_coverage_and_missing_window(self):
        records = [
            {"start_1based": 1, "end_1based": 20, "mutant_logits": [0.0]},
            {"start_1based": 2, "end_1based": 21, "mutant_logits": [5.0]},
        ]
        a = self.api.aggregate_signed_sensitivity(21, 3.0, records)
        self.assertAlmostEqual(a["signed_support"][0], 0.15)
        self.assertAlmostEqual(a["signed_support"][1], 0.025)
        self.assertAlmostEqual(a["signed_support"][-1], -0.1)
        with self.assertRaises(ValueError):
            self.api.aggregate_signed_sensitivity(21, 3.0, records[:1])

    def test_explanation_is_annotation_free_and_full_repeat_budget(self):
        self.assertTrue(hasattr(self.api, "explain"), "MAIA runner missing")
        result = self.api.explain(
            "AC", lambda seqs: [sum(ord(c) for c in s) / 100 for s in seqs]
        )
        self.assertEqual(len(result["ledger"]), 10)
        self.assertEqual(len(result["aggregate"]["signed_support"]), 2)
        self.assertEqual(result["protocol"]["group_seeds"], [1001, 2001])
