import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from heuristic import addr2shard  # noqa: E402
from profile_iot_window import profile_iot_window  # noqa: E402


class IoTWindowProfileTest(unittest.TestCase):
    def test_profiles_requested_owner_window_and_capacity_ceiling(self):
        with TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "sidecar.csv"
            owners = [
                "0x" + "1" * 40,
                "0x" + "2" * 40,
                "0x" + "2" * 40,
                "0x" + "2" * 40,
            ]
            with sidecar.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "tx_index",
                        "time",
                        "from_address",
                        "device_label",
                        "device_mac",
                        "relation_type",
                    ],
                )
                writer.writeheader()
                for idx, owner in enumerate(owners):
                    writer.writerow(
                        {
                            "tx_index": idx,
                            "time": f"2017-01-01 00:00:0{idx}",
                            "from_address": owner,
                            "device_label": "Camera",
                            "device_mac": f"00:00:00:00:00:{idx:02d}",
                            "relation_type": "device_to_cloud",
                        }
                    )

            result = profile_iot_window(
                sidecar_path=sidecar,
                start_tx=1,
                max_txs=3,
                shards=16,
                block_size=1000,
                block_interval_ms=5000,
            )

        hot_sid = addr2shard(owners[1], 16)
        self.assertEqual(result["owner_shard_loads"][hot_sid], 3)
        self.assertAlmostEqual(result["owner_max_load_share"], 1.0)
        self.assertAlmostEqual(result["per_shard_capacity_tps"], 200.0)
        self.assertAlmostEqual(result["owner_anchor_tps_ceiling"], 200.0)
        self.assertEqual(result["start_tx"], 1)
        self.assertEqual(result["end_tx_exclusive"], 4)

    def test_rejects_incomplete_window(self):
        with TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "sidecar.csv"
            sidecar.write_text(
                "tx_index,from_address\n0,0x1111111111111111111111111111111111111111\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "window incomplete"):
                profile_iot_window(
                    sidecar_path=sidecar,
                    start_tx=0,
                    max_txs=2,
                    shards=16,
                    block_size=1000,
                    block_interval_ms=5000,
                )


if __name__ == "__main__":
    unittest.main()
