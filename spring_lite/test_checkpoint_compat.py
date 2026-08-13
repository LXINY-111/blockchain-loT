import unittest

from checkpoint_compat import checkpoint_config_mismatches


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_reports_missing_and_different_values(self):
        mismatches = checkpoint_config_mismatches(
            {"candidate_top_k": 8, "iot_cstr_weight": 0.55},
            {
                "candidate_top_k": 7,
                "iot_cstr_weight": 0.55,
                "reward_mode": "iot_dense_balanced",
            },
        )

        self.assertEqual(len(mismatches), 2)
        self.assertIn("candidate_top_k", mismatches[0])
        self.assertIn("reward_mode", mismatches[1])

    def test_float_values_use_tight_numeric_comparison(self):
        self.assertEqual(
            checkpoint_config_mismatches(
                {"iot_balance_weight": 0.30000000001},
                {"iot_balance_weight": 0.30},
            ),
            [],
        )

    def test_old_checkpoint_without_load_semantics_is_rejected(self):
        mismatches = checkpoint_config_mismatches(
            {"candidate_top_k": 7},
            {
                "candidate_top_k": 7,
                "load_semantics_version": "primary_owner_execution_v1",
            },
        )

        self.assertEqual(len(mismatches), 1)
        self.assertIn("load_semantics_version", mismatches[0])


if __name__ == "__main__":
    unittest.main()
