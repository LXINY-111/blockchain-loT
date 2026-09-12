import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_block_run import (  # noqa: E402
    DATASET_WINDOWS,
    build_proposed_config,
    default_model,
    experiment_name,
    proposed_scheme,
)


class Exp8ProposedRunnerTest(unittest.TestCase):
    def test_validation_calibration_config_uses_fixed_validation_window(self):
        window = DATASET_WINDOWS["validation"]
        config = build_proposed_config(
            base_config={},
            seed=7,
            model_path=Path("E:/project_iot/model.pt"),
            exp_test=Path("E:/project_iot/run/expTest"),
            window=window,
            inject_speed=300,
        )

        self.assertEqual(config["SpringMode"], 2)
        self.assertEqual(config["SpringOnlineTrain"], 0)
        self.assertEqual(config["SpringCandidateTopK"], 7)
        self.assertEqual(config["SpringIOTMode"], 1)
        self.assertEqual(config["SpringIOTIdentityMode"], 1)
        self.assertEqual(config["SpringIOTFeatureDim"], 10)
        self.assertEqual(config["DatasetStartTx"], 2_500_000)
        self.assertEqual(config["TotalDataSize"], 444_019)
        self.assertEqual(config["InjectSpeed"], 300)

    def test_test_window_is_fixed_and_inject_speed_is_explicit(self):
        window = DATASET_WINDOWS["test"]
        config = build_proposed_config(
            base_config={},
            seed=17,
            model_path=Path("E:/project_iot/model.pt"),
            exp_test=Path("E:/project_iot/run/expTest"),
            window=window,
            inject_speed=200,
        )

        self.assertEqual(config["DatasetStartTx"], 2_944_019)
        self.assertEqual(config["TotalDataSize"], 2_000_000)
        self.assertEqual(config["InjectSpeed"], 200)
        self.assertEqual(config["SpringRandomSeed"], 17)

    def test_candidate_parameters_drive_config_and_canonical_names(self):
        window = DATASET_WINDOWS["validation"]
        scheme = proposed_scheme(
            candidate_top_k=6,
            iot_cstr_weight=0.50,
            iot_balance_weight=0.35,
        )
        model = default_model(
            seed=17,
            candidate_top_k=6,
            iot_cstr_weight=0.50,
            iot_balance_weight=0.35,
        )
        name = experiment_name(
            scheme=scheme,
            window=window,
            inject_speed=250,
            seed=17,
        )
        config = build_proposed_config(
            base_config={},
            seed=17,
            model_path=model,
            exp_test=Path("E:/project_iot/run/expTest"),
            window=window,
            inject_speed=250,
            candidate_top_k=6,
            iot_cstr_weight=0.50,
            iot_balance_weight=0.35,
        )

        self.assertEqual(scheme, "ppo_top6_w5035_pareto")
        self.assertEqual(
            model.name,
            "ppo_top6_w5035_pareto_16s_seed17.pt",
        )
        self.assertEqual(
            name,
            "ppo_top6_w5035_pareto_16s_validation444k_i250_seed17",
        )
        self.assertEqual(config["SpringCandidateTopK"], 6)
        self.assertEqual(config["SpringIOTCSTRWeight"], 0.50)
        self.assertEqual(config["SpringIOTBalanceWeight"], 0.35)
        self.assertEqual(config["SpringIOTCommCostWeight"], 0.10)
        self.assertEqual(config["SpringIOTHotspotWeight"], 0.05)

    def test_rejects_ambiguous_or_invalid_candidate_parameters(self):
        with self.assertRaisesRegex(ValueError, "candidate_top_k"):
            proposed_scheme(candidate_top_k=17)
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            proposed_scheme(
                candidate_top_k=7,
                iot_cstr_weight=0.55,
                iot_balance_weight=0.25,
            )
        with self.assertRaisesRegex(ValueError, "increments of 0.01"):
            proposed_scheme(
                candidate_top_k=7,
                iot_cstr_weight=0.555,
                iot_balance_weight=0.295,
            )

    def test_custom_experiment_name_must_keep_configuration_prefix(self):
        scheme = proposed_scheme()
        with self.assertRaisesRegex(ValueError, "canonical"):
            experiment_name(
                scheme=scheme,
                window=DATASET_WINDOWS["validation"],
                inject_speed=250,
                seed=7,
                override="ppo_top8_w5530_pareto_wrong_config",
            )

    def test_rejects_non_positive_inject_speed(self):
        with self.assertRaisesRegex(ValueError, "inject_speed"):
            build_proposed_config(
                base_config={},
                seed=7,
                model_path=Path("E:/project_iot/model.pt"),
                exp_test=Path("E:/project_iot/run/expTest"),
                window=DATASET_WINDOWS["validation"],
                inject_speed=0,
            )


if __name__ == "__main__":
    unittest.main()
