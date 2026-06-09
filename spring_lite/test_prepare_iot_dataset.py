import csv
import json
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_iot_dataset import (  # noqa: E402
    build_iot_dataset,
    endpoint_to_eth_address,
    is_business_flow,
    parse_device_from_flow_name,
)


FLOW_HEADER = [
    "time",
    "srcMac",
    "dstMac",
    "ethType",
    "srcIp",
    "dstIp",
    "ipProto",
    "srcPort",
    "dstPort",
    "flowSeqNum",
    "srcNumPackets",
    "dstNumPackets",
    "srcPayloadSize",
    "dstPayloadSize",
    "srcAvgPayloadSize",
    "dstAvgPayloadSize",
    "srcMaxPayloadSize",
    "dstMaxPayloadSize",
    "srcStdDevPayloadSize",
    "dstStdDevPayloadSize",
    "flowDuration",
    "srcAvgInterarrivalTime",
    "dstAvgInterarrivalTime",
    "avgInterarrivalTime",
    "srcStdDevInterarrivalTime",
    "dstStdDevInterarrivalTime",
    "stdDevInterarrivalTime",
    "allMatchedProtocols",
    "protocol",
]


def flow_row(
    time,
    src_ip,
    dst_ip,
    protocol,
    src_payload,
    dst_payload,
    dst_port="443",
    src_packets="1",
    dst_packets="1",
    all_protocols=None,
):
    row = {name: "" for name in FLOW_HEADER}
    row.update(
        {
            "time": time,
            "srcMac": "null",
            "dstMac": "null",
            "ethType": "0x0800",
            "srcIp": src_ip,
            "dstIp": dst_ip,
            "ipProto": "6",
            "srcPort": "40000",
            "dstPort": dst_port,
            "flowSeqNum": "1",
            "srcNumPackets": src_packets,
            "dstNumPackets": dst_packets,
            "srcPayloadSize": src_payload,
            "dstPayloadSize": dst_payload,
            "flowDuration": "42.0",
            "allMatchedProtocols": all_protocols if all_protocols is not None else protocol,
            "protocol": protocol,
        }
    )
    return row


class PrepareIoTDatasetTest(unittest.TestCase):
    def test_parse_device_name_and_mac_from_flow_file(self):
        info = parse_device_from_flow_name("flows/TPLinkSmartPlug_50c7bf005639_flows.csv")

        self.assertEqual(info.device_label, "TPLinkSmartPlug")
        self.assertEqual(info.device_mac, "50:c7:bf:00:56:39")

    def test_endpoint_hash_is_ethereum_like_and_stable(self):
        first = endpoint_to_eth_address("ip:192.168.1.227")
        second = endpoint_to_eth_address("ip:192.168.1.227")

        self.assertEqual(first, second)
        self.assertRegex(first, r"^0x[0-9a-f]{40}$")

    def test_business_filter_rejects_multicast_and_control_protocols(self):
        valid = flow_row(
            "2016-09-30 19:30:02.000000",
            "192.168.1.227",
            "54.254.250.149",
            "tls",
            "10",
            "20",
        )
        multicast = flow_row(
            "2016-09-30 19:30:03.000000",
            "192.168.1.227",
            "239.255.255.250",
            "ssdp",
            "10",
            "20",
        )

        self.assertTrue(is_business_flow(valid))
        self.assertFalse(is_business_flow(multicast))

    def test_build_iot_dataset_outputs_blockemulator_csv_sidecar_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_zip = root / "flows.zip"
            mote_locs = root / "mote_locs.txt"
            connectivity = root / "connectivity.txt"
            output_dir = root / "data_iot"

            mote_locs.write_text("1 21.5 23\n2 24.5 20\n3 19.5 19\n", encoding="utf-8")
            connectivity.write_text(
                "1 1 0.00000000000000000000\n"
                "1 2 0.80000000000000000000\n"
                "2 1 0.70000000000000000000\n"
                "2 3 0.50000000000000000000\n",
                encoding="utf-8",
            )

            valid_early = flow_row(
                "2016-09-30 19:30:01.000000",
                "192.168.1.10",
                "8.8.8.8",
                "tls",
                "0",
                "0",
                src_packets="3",
                dst_packets="4",
            )
            filtered = flow_row(
                "2016-09-30 19:30:02.000000",
                "192.168.1.10",
                "239.255.255.250",
                "ssdp",
                "100",
                "100",
            )
            valid_late = flow_row(
                "2016-09-30 19:30:03.000000",
                "192.168.1.10",
                "54.254.250.149",
                "https",
                "100",
                "50",
            )

            with zipfile.ZipFile(flows_zip, "w") as archive:
                csv_text = root / "sample.csv"
                with csv_text.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FLOW_HEADER, quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                    writer.writerows([valid_late, filtered, valid_early])
                archive.write(csv_text, "flows/TPLinkSmartPlug_50c7bf005639_flows.csv")

            summary = build_iot_dataset(
                flows_zip_path=flows_zip,
                mote_locs_path=mote_locs,
                connectivity_path=connectivity,
                output_dir=output_dir,
                limit=2,
            )

            tx_path = output_dir / "selectedTxs_iot_2.csv"
            sidecar_path = output_dir / "iot_flow_sidecar_2.csv"
            topology_path = output_dir / "iot_device_topology.csv"
            summary_path = output_dir / "iot_dataset_summary.json"

            self.assertTrue(tx_path.exists())
            self.assertTrue(sidecar_path.exists())
            self.assertTrue(topology_path.exists())
            self.assertTrue(summary_path.exists())

            with tx_path.open("r", encoding="utf-8", newline="") as handle:
                tx_rows = list(csv.reader(handle))
            self.assertEqual(len(tx_rows), 2)
            self.assertTrue(all(len(row) == 18 for row in tx_rows))
            self.assertEqual(tx_rows[0][6], "0")
            self.assertEqual(tx_rows[0][7], "0")
            self.assertEqual(tx_rows[0][8], "7")
            self.assertRegex(tx_rows[0][3], r"^0x[0-9a-f]{40}$")
            self.assertRegex(tx_rows[0][4], r"^0x[0-9a-f]{40}$")

            with sidecar_path.open("r", encoding="utf-8", newline="") as handle:
                sidecar_rows = list(csv.DictReader(handle))
            self.assertEqual([row["time"] for row in sidecar_rows], [valid_early["time"], valid_late["time"]])
            self.assertEqual(sidecar_rows[0]["device_label"], "TPLinkSmartPlug")
            self.assertIn("mapped_mote_id", sidecar_rows[0])
            self.assertIn("distance", sidecar_rows[0])
            self.assertIn("link_quality", sidecar_rows[0])
            self.assertEqual(sidecar_rows[0]["tx_batch_id"], "0")

            with summary_path.open("r", encoding="utf-8") as handle:
                persisted_summary = json.load(handle)
            self.assertEqual(summary["selected_flow_count"], 2)
            self.assertEqual(persisted_summary["filtered_flow_count"], 1)
            self.assertEqual(persisted_summary["output_files"]["transactions"], str(tx_path))

    def test_state_object_address_separates_service_and_protocol(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_zip = root / "flows.zip"
            mote_locs = root / "mote_locs.txt"
            connectivity = root / "connectivity.txt"
            output_dir = root / "data_iot"

            mote_locs.write_text("1 21.5 23\n2 24.5 20\n3 19.5 19\n", encoding="utf-8")
            connectivity.write_text("1 2 0.80000000000000000000\n", encoding="utf-8")

            tls_443 = flow_row(
                "2016-09-30 19:30:01.000000",
                "192.168.1.10",
                "54.254.250.149",
                "tls",
                "100",
                "50",
                dst_port="443",
            )
            http_80 = flow_row(
                "2016-09-30 19:30:02.000000",
                "192.168.1.10",
                "54.254.250.149",
                "http",
                "120",
                "20",
                dst_port="80",
            )

            with zipfile.ZipFile(flows_zip, "w") as archive:
                csv_text = root / "sample.csv"
                with csv_text.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FLOW_HEADER, quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                    writer.writerows([tls_443, http_80])
                archive.write(csv_text, "flows/TPLinkSmartPlug_50c7bf005639_flows.csv")

            build_iot_dataset(
                flows_zip_path=flows_zip,
                mote_locs_path=mote_locs,
                connectivity_path=connectivity,
                output_dir=output_dir,
                limit=2,
            )

            with (output_dir / "iot_flow_sidecar_2.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                sidecar_rows = list(csv.DictReader(handle))

            self.assertEqual(len({row["to_address"] for row in sidecar_rows}), 2)

    def test_limit_zero_writes_all_valid_flows_and_filters_anomalous_timestamps(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_zip = root / "flows.zip"
            mote_locs = root / "mote_locs.txt"
            connectivity = root / "connectivity.txt"
            output_dir = root / "data_iot"

            mote_locs.write_text("1 21.5 23\n2 24.5 20\n3 19.5 19\n", encoding="utf-8")
            connectivity.write_text("1 2 0.80000000000000000000\n", encoding="utf-8")

            anomaly = flow_row(
                "1970-01-01 08:57:08.620602",
                "192.168.1.10",
                "54.254.250.149",
                "tls",
                "100",
                "50",
            )
            valid_early = flow_row(
                "2016-09-30 19:30:01.000000",
                "192.168.1.10",
                "8.8.8.8",
                "tls",
                "0",
                "0",
                src_packets="3",
                dst_packets="4",
            )
            valid_late = flow_row(
                "2016-09-30 19:30:03.000000",
                "192.168.1.10",
                "54.254.250.149",
                "https",
                "100",
                "50",
            )

            with zipfile.ZipFile(flows_zip, "w") as archive:
                csv_text = root / "sample.csv"
                with csv_text.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FLOW_HEADER, quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                    writer.writerows([valid_late, anomaly, valid_early])
                archive.write(csv_text, "flows/TPLinkSmartPlug_50c7bf005639_flows.csv")

            summary = build_iot_dataset(
                flows_zip_path=flows_zip,
                mote_locs_path=mote_locs,
                connectivity_path=connectivity,
                output_dir=output_dir,
                limit=0,
            )

            sidecar_path = output_dir / "iot_flow_sidecar_full.csv"
            with sidecar_path.open("r", encoding="utf-8", newline="") as handle:
                sidecar_rows = list(csv.DictReader(handle))

            self.assertEqual(summary["selected_flow_count"], 2)
            self.assertEqual(summary["invalid_time_flow_count"], 1)
            self.assertEqual([row["time"] for row in sidecar_rows], [valid_early["time"], valid_late["time"]])


if __name__ == "__main__":
    unittest.main()
