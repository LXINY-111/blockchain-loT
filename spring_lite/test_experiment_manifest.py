import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_experiment_manifest import build_manifest  # noqa: E402


class ExperimentManifestTest(unittest.TestCase):
    def make_args(self, root: Path, configured_start: int = 10):
        dataset = root / "dataset.csv"
        sidecar = root / "sidecar.csv"
        model = root / "model.pt"
        params = root / "paramsConfig.json"
        dataset.write_text("dataset", encoding="utf-8")
        sidecar.write_text("sidecar", encoding="utf-8")
        model.write_bytes(b"model")
        params.write_text(
            json.dumps({"DatasetStartTx": configured_start, "TotalDataSize": 20}),
            encoding="utf-8",
        )
        return argparse.Namespace(
            experiment="unit_test",
            phase="block_eval",
            seed=7,
            shards=16,
            dataset=str(dataset),
            sidecar=str(sidecar),
            model=str(model),
            params=str(params),
            start_tx=10,
            max_txs=20,
            skip_large_file_hash=True,
            model_not_applicable=False,
        )

    def test_manifest_records_window_hashes_and_params_snapshot(self):
        with TemporaryDirectory() as tmp:
            manifest = build_manifest(self.make_args(Path(tmp)))

        self.assertEqual(manifest["data_window"]["start_tx"], 10)
        self.assertEqual(manifest["data_window"]["end_tx_exclusive"], 30)
        self.assertIsNone(manifest["files"]["dataset"]["sha256"])
        self.assertEqual(len(manifest["files"]["model"]["sha256"]), 64)
        self.assertEqual(manifest["params_snapshot"]["TotalDataSize"], 20)

    def test_block_manifest_rejects_params_window_mismatch(self):
        with TemporaryDirectory() as tmp:
            args = self.make_args(Path(tmp), configured_start=11)
            with self.assertRaisesRegex(ValueError, "data window mismatch"):
                build_manifest(args)

    def test_manifest_allows_model_to_be_not_applicable(self):
        with TemporaryDirectory() as tmp:
            args = self.make_args(Path(tmp))
            args.model_not_applicable = True
            Path(args.model).unlink()

            manifest = build_manifest(args)

        self.assertIsNone(manifest["files"]["model"])


if __name__ == "__main__":
    unittest.main()
