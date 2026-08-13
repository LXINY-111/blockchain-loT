import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_env import load_iot_transactions, load_transactions  # noqa: E402
from train_offline import data_windows_overlap  # noqa: E402


class DatasetWindowTest(unittest.TestCase):
    def write_fixture(self, root: Path, bad_sidecar_index: bool = False):
        tx_csv = root / "tx.csv"
        sidecar_csv = root / "sidecar.csv"
        from_addresses = ["0x" + str(idx + 1) * 40 for idx in range(5)]
        to_addresses = ["0x" + chr(ord("a") + idx) * 40 for idx in range(5)]

        with tx_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for idx in range(5):
                if idx == 1:
                    writer.writerow(["invalid", "row"])
                row = [""] * 18
                row[3] = from_addresses[idx]
                row[4] = to_addresses[idx]
                row[6] = "0"
                row[7] = "0"
                row[8] = "1"
                writer.writerow(row)

        fieldnames = [
            "tx_index",
            "device_label",
            "device_mac",
            "srcIp",
            "dstIp",
            "srcPort",
            "dstPort",
            "protocol",
            "srcNumPackets",
            "dstNumPackets",
            "srcPayloadSize",
            "dstPayloadSize",
            "from_address",
            "to_address",
            "distance",
            "link_quality",
        ]
        with sidecar_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(5):
                writer.writerow(
                    {
                        "tx_index": 99 if bad_sidecar_index and idx == 2 else idx,
                        "device_label": "Camera",
                        "device_mac": f"aa:bb:cc:dd:ee:{idx:02d}",
                        "srcIp": f"192.168.1.{idx + 2}",
                        "dstIp": "8.8.8.8",
                        "srcPort": "40000",
                        "dstPort": "443",
                        "protocol": "tls",
                        "srcNumPackets": "2",
                        "dstNumPackets": "2",
                        "srcPayloadSize": "100",
                        "dstPayloadSize": "200",
                        "from_address": from_addresses[idx],
                        "to_address": to_addresses[idx],
                        "distance": "25",
                        "link_quality": "0.5",
                    }
                )
        return tx_csv, sidecar_csv, from_addresses, to_addresses

    def test_raw_window_counts_only_valid_transactions(self):
        with TemporaryDirectory() as tmp:
            tx_csv, _sidecar, from_addresses, _to_addresses = self.write_fixture(Path(tmp))
            txs = load_transactions(tx_csv, start_tx=2, max_txs=2)

        self.assertEqual([tx.sender for tx in txs], [addr[2:] for addr in from_addresses[2:4]])

    def test_iot_window_aligns_sidecar_and_resets_to_requested_slice(self):
        with TemporaryDirectory() as tmp:
            tx_csv, sidecar_csv, _from_addresses, to_addresses = self.write_fixture(Path(tmp))
            txs = load_iot_transactions(
                tx_csv,
                sidecar_csv,
                start_tx=2,
                max_txs=2,
            )

        self.assertEqual([tx.sender for tx in txs], to_addresses[2:4])
        self.assertEqual(len(txs), 2)

    def test_iot_window_rejects_non_contiguous_sidecar_index(self):
        with TemporaryDirectory() as tmp:
            tx_csv, sidecar_csv, _from_addresses, _to_addresses = self.write_fixture(
                Path(tmp),
                bad_sidecar_index=True,
            )
            with self.assertRaisesRegex(ValueError, "index mismatch"):
                load_iot_transactions(tx_csv, sidecar_csv, start_tx=2, max_txs=1)

    def test_window_bounds_and_overlap_rules(self):
        with TemporaryDirectory() as tmp:
            tx_csv, _sidecar, _from_addresses, _to_addresses = self.write_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "start_tx"):
                load_transactions(tx_csv, start_tx=-1, max_txs=1)

        self.assertFalse(data_windows_overlap(0, 3, 3, 2))
        self.assertTrue(data_windows_overlap(0, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
