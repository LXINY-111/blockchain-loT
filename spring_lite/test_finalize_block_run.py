import json
import unittest

from finalize_exp8_block_run import (
    require_executed_state_test,
    validate_analysis_integrity,
)


class FinalizeBlockRunTest(unittest.TestCase):
    def valid_analysis(self):
        return {
            "periods": {"all_active": {"effective_total": 3.0}},
            "latency_details": {
                "row_count": 3,
                "valid_latency_count": 3,
                "invalid_latency_count": 0,
                "unique_tx_hash_count": 3,
                "duplicate_tx_hash_count": 0,
                "missing_tx_hash_count": 0,
            },
        }

    def test_integrity_requires_exact_completion_and_latency_coverage(self):
        integrity = validate_analysis_integrity(
            self.valid_analysis(),
            {"total_data_size": 3},
        )

        self.assertEqual(integrity["completion_ratio"], 1.0)
        self.assertEqual(integrity["latency_coverage_ratio"], 1.0)
        self.assertEqual(integrity["unique_transaction_ratio"], 1.0)

    def test_integrity_rejects_missing_latency_and_excess_transactions(self):
        missing = self.valid_analysis()
        missing["latency_details"]["row_count"] = 1
        missing["latency_details"]["valid_latency_count"] = 1
        missing["latency_details"]["unique_tx_hash_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "completeness"):
            validate_analysis_integrity(missing, {"total_data_size": 3})

        excess = self.valid_analysis()
        excess["periods"]["all_active"]["effective_total"] = 4.0
        with self.assertRaisesRegex(RuntimeError, "effective_total"):
            validate_analysis_integrity(excess, {"total_data_size": 3})

    def test_state_test_must_execute_and_must_not_skip(self):
        passed = json.dumps(
            {"Action": "pass", "Test": "TestFinalResult"}
        )
        require_executed_state_test(passed)

        skipped = json.dumps(
            {"Action": "skip", "Test": "TestFinalResult"}
        )
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            require_executed_state_test(skipped)


if __name__ == "__main__":
    unittest.main()
