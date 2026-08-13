import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_exp8_baseline import (  # noqa: E402
    BASELINE_PROFILES,
    RESULTS_ROOT,
    TEST_MAX_TXS,
    TEST_START_TX,
    build_baseline_config,
)
from config import EXPERIMENT_ROOT  # noqa: E402


class Exp8BaselineRunnerTest(unittest.TestCase):
    def test_all_runners_share_result9_root(self):
        self.assertEqual(RESULTS_ROOT, EXPERIMENT_ROOT)
        self.assertEqual(RESULTS_ROOT.name, "实验结果9")

    def test_six_core_baselines_and_one_ablation_are_declared(self):
        baseline_count = sum(
            profile.role == "baseline" for profile in BASELINE_PROFILES.values()
        )
        ablation_count = sum(
            profile.role == "ablation" for profile in BASELINE_PROFILES.values()
        )

        self.assertEqual(baseline_count, 6)
        self.assertEqual(ablation_count, 1)
        self.assertEqual(BASELINE_PROFILES["anchoronly"].spring_mode, 5)
        self.assertEqual(BASELINE_PROFILES["nsshard_adapted"].spring_mode, 6)

    def test_every_profile_uses_the_same_iot_identity_and_test_window(self):
        exp_test = Path("E:/project_iot/placeholder/expTest")
        model = Path("E:/project_iot/model.pt")
        for profile in BASELINE_PROFILES.values():
            with self.subTest(scheme=profile.scheme):
                config = build_baseline_config(
                    {},
                    profile,
                    seed=7,
                    exp_test=exp_test,
                    model_path=model if profile.requires_model else None,
                )
                self.assertEqual(config["SpringMode"], profile.spring_mode)
                self.assertEqual(config["SpringOnlineTrain"], 0)
                self.assertEqual(config["SpringIOTMode"], 0)
                self.assertEqual(config["SpringIOTIdentityMode"], 1)
                self.assertEqual(config["SpringIOTFeatureDim"], 0)
                self.assertEqual(config["SpringCandidateTopK"], 0)
                self.assertEqual(config["DatasetStartTx"], TEST_START_TX)
                self.assertEqual(config["TotalDataSize"], TEST_MAX_TXS)

    def test_only_original_spring_requires_a_frozen_model(self):
        requiring_model = [
            profile.scheme
            for profile in BASELINE_PROFILES.values()
            if profile.requires_model
        ]

        self.assertEqual(requiring_model, ["original_spring_ppo"])

    def test_baseline_calibration_can_use_validation_window_and_fixed_rate(self):
        profile = BASELINE_PROFILES["hash"]
        config = build_baseline_config(
            {},
            profile,
            seed=7,
            exp_test=Path("E:/project_iot/placeholder/expTest"),
            model_path=None,
            start_tx=2_500_000,
            max_txs=444_019,
            inject_speed=300,
        )

        self.assertEqual(config["DatasetStartTx"], 2_500_000)
        self.assertEqual(config["TotalDataSize"], 444_019)
        self.assertEqual(config["InjectSpeed"], 300)


if __name__ == "__main__":
    unittest.main()
