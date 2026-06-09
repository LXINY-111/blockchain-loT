import argparse
import csv
import hashlib
import heapq
import io
import ipaddress
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


CONTROL_PROTOCOLS = {
    "arp",
    "dhcp",
    "dhcpv6",
    "icmp",
    "icmpv6",
    "igmp",
    "llmnr",
    "mdns",
    "ssdp",
}

VALID_FLOW_YEARS = {2016, 2017}

SIDECAR_FIELDS = [
    "tx_index",
    "time",
    "device_label",
    "device_mac",
    "srcIp",
    "dstIp",
    "srcPort",
    "dstPort",
    "protocol",
    "state_object_key",
    "srcNumPackets",
    "dstNumPackets",
    "srcPayloadSize",
    "dstPayloadSize",
    "flowDuration",
    "from_address",
    "to_address",
    "mapped_mote_id",
    "mapped_x",
    "mapped_y",
    "related_mote_id",
    "related_x",
    "related_y",
    "distance",
    "link_quality",
    "tx_batch_id",
]

TOPOLOGY_FIELDS = [
    "device_label",
    "device_mac",
    "flow_file",
    "mapped_mote_id",
    "mapped_x",
    "mapped_y",
]


@dataclass(frozen=True)
class DeviceInfo:
    device_label: str
    device_mac: str
    flow_file: str


@dataclass(frozen=True)
class MoteLocation:
    mote_id: int
    x: float
    y: float


@dataclass(frozen=True)
class SelectedFlow:
    time_key: float
    sequence: int
    flow_file: str
    row: Mapping[str, str]


def parse_device_from_flow_name(flow_name: str) -> DeviceInfo:
    """从 UNSW flow 文件名中解析设备标签和 MAC 地址。"""
    file_name = Path(flow_name).name
    stem = file_name
    if stem.endswith("_flows.csv"):
        stem = stem[: -len("_flows.csv")]
    elif stem.endswith(".csv"):
        stem = stem[: -len(".csv")]

    match = re.match(r"^(?P<label>.+)_(?P<mac>[0-9a-fA-F]{12})$", stem)
    if not match:
        return DeviceInfo(device_label=stem, device_mac="", flow_file=flow_name)

    mac = match.group("mac").lower()
    formatted_mac = ":".join(mac[index : index + 2] for index in range(0, 12, 2))
    return DeviceInfo(
        device_label=match.group("label"),
        device_mac=formatted_mac,
        flow_file=flow_name,
    )


def endpoint_to_eth_address(endpoint: str) -> str:
    """把 IP/MAC 端点稳定映射成 BlockEmulator 可读的以太坊风格地址。"""
    normalized = (endpoint or "unknown").strip().lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return "0x" + digest


def device_sender_key(device: DeviceInfo) -> str:
    label = clean_cell(device.device_label).lower() or "unknown_device"
    mac = clean_cell(device.device_mac).lower() or "unknown_mac"
    return f"iot_device:{label}|mac:{mac}"


def state_object_key(device: DeviceInfo, row: Mapping[str, str], protocol: Optional[str] = None) -> str:
    label = clean_cell(device.device_label).lower() or "unknown_device"
    dst_ip = clean_cell(row.get("dstIp", "")).lower() or "unknown_dst"
    dst_port = clean_cell(row.get("dstPort", "")) or "any"
    proto = (protocol or normalized_protocol(row)).lower() or "none"
    return f"iot_state:device:{label}|dst:{dst_ip}|service:{dst_port}|proto:{proto}"


def is_business_flow(row: Mapping[str, str]) -> bool:
    """过滤广播、多播和基础控制协议，仅保留适合作为业务交易的通信流。"""
    src_endpoint = endpoint_identity(row, "src")
    dst_endpoint = endpoint_identity(row, "dst")
    if src_endpoint is None or dst_endpoint is None:
        return False

    if protocol_tokens(row) & CONTROL_PROTOCOLS:
        return False

    for key in ("srcIp", "dstIp"):
        if is_multicast_or_broadcast_ip(row.get(key, "")):
            return False
    for key in ("srcMac", "dstMac"):
        if is_multicast_or_broadcast_mac(row.get(key, "")):
            return False

    return True


def build_iot_dataset(
    flows_zip_path: Path,
    mote_locs_path: Path,
    connectivity_path: Path,
    output_dir: Path,
    limit: int = 300_000,
    tx_batch_size: int = 1000,
) -> Dict[str, object]:
    """将 IoT flow 转成 BlockEmulator 交易 CSV，并生成 PPO/MDP 可用的 sidecar。"""
    flows_zip_path = Path(flows_zip_path)
    mote_locs_path = Path(mote_locs_path)
    connectivity_path = Path(connectivity_path)
    output_dir = Path(output_dir)

    validate_inputs(flows_zip_path, mote_locs_path, connectivity_path, limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    mote_locs = load_mote_locs(mote_locs_path)
    connectivity = load_connectivity(connectivity_path)

    total_flow_count = 0
    non_business_flow_count = 0
    filtered_flow_count = 0
    invalid_time_flow_count = 0
    protocol_seen = Counter()
    protocol_selected = Counter()
    device_selected_counts = Counter()
    address_mapping: Dict[str, str] = {}
    selected_state_objects = set()
    selected_from_addresses = set()
    selected_to_addresses = set()

    with zipfile.ZipFile(flows_zip_path, "r") as archive:
        flow_entries = sorted(
            [entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")],
            key=lambda entry: entry.filename,
        )
        device_infos = {
            entry.filename: parse_device_from_flow_name(entry.filename) for entry in flow_entries
        }
        device_topology = assign_device_topology(device_infos.values(), mote_locs)

        # 只保留时间最早的 limit 条有效 flow，避免把 1GB+ 数据全部常驻内存。
        selected_heap: List[Tuple[float, int, SelectedFlow]] = []
        selected_all: List[SelectedFlow] = []
        sequence = 0
        for entry in flow_entries:
            with archive.open(entry, "r") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                reader = csv.DictReader(text)
                for row in reader:
                    total_flow_count += 1
                    protocol_seen[normalized_protocol(row)] += 1
                    if not is_business_flow(row):
                        non_business_flow_count += 1
                        filtered_flow_count += 1
                        continue

                    time_key = parse_time_key(row.get("time", ""))
                    if time_key is None:
                        invalid_time_flow_count += 1
                        filtered_flow_count += 1
                        continue

                    selected = SelectedFlow(
                        time_key=time_key,
                        sequence=sequence,
                        flow_file=entry.filename,
                        row=dict(row),
                    )
                    if limit == 0:
                        selected_all.append(selected)
                    else:
                        push_earliest(selected_heap, selected, limit)
                    sequence += 1

        if limit == 0:
            selected_flows = selected_all
        else:
            selected_flows = [
                selected for _negative_time, _negative_sequence, selected in selected_heap
            ]
        selected_flows.sort(key=lambda item: (item.time_key, item.sequence))

    suffix = format_limit_suffix(limit)
    tx_path = output_dir / f"selectedTxs_iot_{suffix}.csv"
    sidecar_path = output_dir / f"iot_flow_sidecar_{suffix}.csv"
    topology_path = output_dir / "iot_device_topology.csv"
    summary_path = output_dir / "iot_dataset_summary.json"

    with tx_path.open("w", encoding="utf-8", newline="") as tx_handle, sidecar_path.open(
        "w", encoding="utf-8", newline=""
    ) as sidecar_handle:
        tx_writer = csv.writer(tx_handle)
        sidecar_writer = csv.DictWriter(sidecar_handle, fieldnames=SIDECAR_FIELDS)
        sidecar_writer.writeheader()

        for tx_index, selected in enumerate(selected_flows):
            row = selected.row
            device = device_infos[selected.flow_file]
            protocol = normalized_protocol(row)
            state_key = state_object_key(device, row, protocol)
            from_endpoint = device_sender_key(device)
            to_endpoint = state_key
            from_address = address_mapping.setdefault(
                from_endpoint, endpoint_to_eth_address(from_endpoint)
            )
            to_address = address_mapping.setdefault(to_endpoint, endpoint_to_eth_address(to_endpoint))
            value = flow_value(row)

            tx_writer.writerow(blockemulator_tx_row(from_address, to_address, value))

            mapped = device_topology[device.flow_file]
            related_endpoint = choose_related_endpoint(row)
            related = related_mote_for_endpoint(related_endpoint, mapped.mote_id, mote_locs)
            distance = euclidean_distance(mapped, related)
            link_quality = lookup_link_quality(connectivity, mapped.mote_id, related.mote_id, distance)

            protocol_selected[protocol] += 1
            device_selected_counts[device.device_label] += 1
            selected_state_objects.add(state_key)
            selected_from_addresses.add(from_address)
            selected_to_addresses.add(to_address)

            sidecar_writer.writerow(
                {
                    "tx_index": tx_index,
                    "time": clean_cell(row.get("time", "")),
                    "device_label": device.device_label,
                    "device_mac": device.device_mac,
                    "srcIp": clean_cell(row.get("srcIp", "")),
                    "dstIp": clean_cell(row.get("dstIp", "")),
                    "srcPort": clean_cell(row.get("srcPort", "")),
                    "dstPort": clean_cell(row.get("dstPort", "")),
                    "protocol": protocol,
                    "state_object_key": state_key,
                    "srcNumPackets": clean_cell(row.get("srcNumPackets", "")),
                    "dstNumPackets": clean_cell(row.get("dstNumPackets", "")),
                    "srcPayloadSize": clean_cell(row.get("srcPayloadSize", "")),
                    "dstPayloadSize": clean_cell(row.get("dstPayloadSize", "")),
                    "flowDuration": clean_cell(row.get("flowDuration", "")),
                    "from_address": from_address,
                    "to_address": to_address,
                    "mapped_mote_id": mapped.mote_id,
                    "mapped_x": format_float(mapped.x),
                    "mapped_y": format_float(mapped.y),
                    "related_mote_id": related.mote_id,
                    "related_x": format_float(related.x),
                    "related_y": format_float(related.y),
                    "distance": format_float(distance),
                    "link_quality": format_float(link_quality),
                    "tx_batch_id": tx_index // tx_batch_size,
                }
            )

    write_topology(topology_path, device_infos.values(), device_topology)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "flows_zip": str(flows_zip_path),
            "mote_locs": str(mote_locs_path),
            "connectivity": str(connectivity_path),
        },
        "output_files": {
            "transactions": str(tx_path),
            "sidecar": str(sidecar_path),
            "topology": str(topology_path),
            "summary": str(summary_path),
        },
        "limit": limit,
        "tx_batch_size": tx_batch_size,
        "flow_file_count": len(device_infos),
        "device_count": len(device_infos),
        "total_flow_count": total_flow_count,
        "non_business_flow_count": non_business_flow_count,
        "invalid_time_flow_count": invalid_time_flow_count,
        "filtered_flow_count": filtered_flow_count,
        "valid_flow_count": total_flow_count - filtered_flow_count,
        "selected_flow_count": len(selected_flows),
        "selected_state_object_count": len(selected_state_objects),
        "selected_from_address_count": len(selected_from_addresses),
        "selected_to_address_count": len(selected_to_addresses),
        "address_grain": {
            "from_address": "hash(device_label, device_mac)",
            "to_address": "hash(device_label, dstIp, dstPort, protocol)",
            "state_object_key": "device_label + dstIp + dstPort + protocol",
        },
        "filter_rules": {
            "excluded_protocols": sorted(CONTROL_PROTOCOLS),
            "valid_time_years": sorted(VALID_FLOW_YEARS),
            "excluded_ip_patterns": [
                "IPv4 multicast 224.0.0.0/4",
                "IPv4 broadcast 255.255.255.255",
                "IPv4 unspecified 0.0.0.0",
                "IPv6 multicast ff00::/8",
            ],
            "excluded_mac_patterns": [
                "ff:ff:ff:ff:ff:ff",
                "multicast MAC with low bit set in the first octet",
            ],
            "missing_endpoint_policy": "drop rows without usable src/dst IP or MAC",
        },
        "protocol_distribution_seen": dict(sorted(protocol_seen.items())),
        "protocol_distribution_selected": dict(sorted(protocol_selected.items())),
        "selected_flow_count_by_device": dict(sorted(device_selected_counts.items())),
        "address_mapping": dict(sorted(address_mapping.items())),
        "topology_augmentation": {
            "method": "UNSW flow devices are deterministically mapped to Intel Berkeley mote coordinates; peer endpoints are hashed to related motes.",
            "mote_count": len(mote_locs),
            "connectivity_edge_count": len(connectivity),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def validate_inputs(
    flows_zip_path: Path, mote_locs_path: Path, connectivity_path: Path, limit: int
) -> None:
    for path in (flows_zip_path, mote_locs_path, connectivity_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if limit < 0:
        raise ValueError("limit must be zero for all valid flows or a positive row cap")


def load_mote_locs(path: Path) -> Dict[int, MoteLocation]:
    motes: Dict[int, MoteLocation] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            mote_id = int(parts[0])
            motes[mote_id] = MoteLocation(mote_id=mote_id, x=float(parts[1]), y=float(parts[2]))
    if not motes:
        raise ValueError(f"no mote locations found in {path}")
    return dict(sorted(motes.items()))


def load_connectivity(path: Path) -> Dict[Tuple[int, int], float]:
    connectivity: Dict[Tuple[int, int], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            src = int(parts[0])
            dst = int(parts[1])
            connectivity[(src, dst)] = float(parts[2])
    return connectivity


def assign_device_topology(
    devices: Iterable[DeviceInfo], motes: Mapping[int, MoteLocation]
) -> Dict[str, MoteLocation]:
    mote_ids = list(motes.keys())
    available = list(mote_ids)
    assignments: Dict[str, MoteLocation] = {}

    for device in sorted(devices, key=lambda item: (item.device_label, item.device_mac)):
        seed = stable_int(f"{device.device_label}|{device.device_mac}|{device.flow_file}")
        if available:
            index = seed % len(available)
            mote_id = available.pop(index)
        else:
            mote_id = mote_ids[seed % len(mote_ids)]
        assignments[device.flow_file] = motes[mote_id]

    return assignments


def write_topology(
    path: Path,
    devices: Iterable[DeviceInfo],
    device_topology: Mapping[str, MoteLocation],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOPOLOGY_FIELDS)
        writer.writeheader()
        for device in sorted(devices, key=lambda item: (item.device_label, item.device_mac)):
            mote = device_topology[device.flow_file]
            writer.writerow(
                {
                    "device_label": device.device_label,
                    "device_mac": device.device_mac,
                    "flow_file": device.flow_file,
                    "mapped_mote_id": mote.mote_id,
                    "mapped_x": format_float(mote.x),
                    "mapped_y": format_float(mote.y),
                }
            )


def push_earliest(
    heap: List[Tuple[float, int, SelectedFlow]], selected: SelectedFlow, limit: int
) -> None:
    item = (-selected.time_key, -selected.sequence, selected)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return

    latest_time = -heap[0][0]
    latest_sequence = -heap[0][1]
    if (selected.time_key, selected.sequence) < (latest_time, latest_sequence):
        heapq.heapreplace(heap, item)


def parse_time_key(value: str) -> Optional[float]:
    text = clean_cell(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if parsed.year not in VALID_FLOW_YEARS:
            return None
        return parsed.timestamp()
    except ValueError:
        return None


def format_limit_suffix(limit: int) -> str:
    if limit == 0:
        return "full"
    if limit % 1000 == 0:
        return f"{limit // 1000}K"
    return str(limit)


def blockemulator_tx_row(from_address: str, to_address: str, value: int) -> List[str]:
    row = [""] * 18
    row[3] = from_address
    row[4] = to_address
    row[6] = "0"
    row[7] = "0"
    row[8] = str(value)
    return row


def flow_value(row: Mapping[str, str]) -> int:
    payload_total = number_to_int(row.get("srcPayloadSize", "")) + number_to_int(
        row.get("dstPayloadSize", "")
    )
    if payload_total > 0:
        return payload_total

    packet_total = number_to_int(row.get("srcNumPackets", "")) + number_to_int(
        row.get("dstNumPackets", "")
    )
    if packet_total > 0:
        return packet_total
    return 1


def number_to_int(value: str) -> int:
    try:
        return int(round(float(clean_cell(value))))
    except (TypeError, ValueError):
        return 0


def endpoint_identity(row: Mapping[str, str], side: str) -> Optional[str]:
    ip_value = clean_cell(row.get(f"{side}Ip", ""))
    if usable_endpoint_value(ip_value):
        return "ip:" + ip_value.lower()

    mac_value = normalize_mac(row.get(f"{side}Mac", ""))
    if usable_endpoint_value(mac_value):
        return "mac:" + mac_value

    return None


def choose_related_endpoint(row: Mapping[str, str]) -> str:
    src_endpoint = endpoint_identity(row, "src") or "unknown:src"
    dst_endpoint = endpoint_identity(row, "dst") or "unknown:dst"
    src_ip = clean_cell(row.get("srcIp", ""))
    dst_ip = clean_cell(row.get("dstIp", ""))

    if is_private_ip(src_ip) and not is_private_ip(dst_ip):
        return dst_endpoint
    if is_private_ip(dst_ip) and not is_private_ip(src_ip):
        return src_endpoint
    return dst_endpoint


def related_mote_for_endpoint(
    endpoint: str, mapped_mote_id: int, motes: Mapping[int, MoteLocation]
) -> MoteLocation:
    mote_ids = list(motes.keys())
    index = stable_int(endpoint) % len(mote_ids)
    mote_id = mote_ids[index]
    if mote_id == mapped_mote_id and len(mote_ids) > 1:
        mote_id = mote_ids[(index + 1) % len(mote_ids)]
    return motes[mote_id]


def lookup_link_quality(
    connectivity: Mapping[Tuple[int, int], float], src: int, dst: int, distance: float
) -> float:
    if (src, dst) in connectivity:
        return connectivity[(src, dst)]
    if (dst, src) in connectivity:
        return connectivity[(dst, src)]
    return 1.0 / (1.0 + distance)


def euclidean_distance(first: MoteLocation, second: MoteLocation) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def normalized_protocol(row: Mapping[str, str]) -> str:
    protocol = clean_cell(row.get("protocol", "")).lower()
    if protocol:
        return protocol
    tokens = sorted(protocol_tokens(row))
    return tokens[0] if tokens else "none"


def protocol_tokens(row: Mapping[str, str]) -> set:
    tokens = set()
    for key in ("protocol", "allMatchedProtocols"):
        raw_value = clean_cell(row.get(key, "")).lower()
        for token in re.split(r"[|,;/\s]+", raw_value):
            token = token.strip()
            if token and token != "none":
                tokens.add(token)
    return tokens


def is_multicast_or_broadcast_ip(value: str) -> bool:
    text = clean_cell(value)
    if not usable_endpoint_value(text):
        return False
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    return address.is_multicast or address.is_unspecified or str(address) == "255.255.255.255"


def is_private_ip(value: str) -> bool:
    text = clean_cell(value)
    if not usable_endpoint_value(text):
        return False
    try:
        return ipaddress.ip_address(text).is_private
    except ValueError:
        return False


def is_multicast_or_broadcast_mac(value: str) -> bool:
    mac = normalize_mac(value)
    if not usable_endpoint_value(mac):
        return False
    compact = mac.replace(":", "")
    if compact == "ffffffffffff":
        return True
    try:
        first_octet = int(compact[:2], 16)
    except ValueError:
        return False
    return bool(first_octet & 1)


def normalize_mac(value: str) -> str:
    text = clean_cell(value).lower().replace("-", ":")
    if not usable_endpoint_value(text):
        return ""

    compact = re.sub(r"[^0-9a-f]", "", text)
    if len(compact) == 12:
        return ":".join(compact[index : index + 2] for index in range(0, 12, 2))
    return text


def usable_endpoint_value(value: str) -> bool:
    text = clean_cell(value).lower()
    return text not in {"", "null", "none", "nan", "0.0.0.0", "::"}


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"')


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16)


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare UNSW-IoTraffic flows for BlockEmulator + SPRING/PPO IoT experiments."
    )
    parser.add_argument("--flows-zip", type=Path, default=Path(r"E:\flows.zip"))
    parser.add_argument("--mote-locs", type=Path, default=Path(r"E:\mote_locs.txt"))
    parser.add_argument("--connectivity", type=Path, default=Path(r"E:\connectivity.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_iot"))
    parser.add_argument("--limit", type=int, default=300_000)
    parser.add_argument("--tx-batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_iot_dataset(
        flows_zip_path=args.flows_zip,
        mote_locs_path=args.mote_locs,
        connectivity_path=args.connectivity,
        output_dir=args.output_dir,
        limit=args.limit,
        tx_batch_size=args.tx_batch_size,
    )
    printable_summary = {
        "transactions": summary["output_files"]["transactions"],
        "sidecar": summary["output_files"]["sidecar"],
        "topology": summary["output_files"]["topology"],
        "summary": summary["output_files"]["summary"],
        "total_flow_count": summary["total_flow_count"],
        "filtered_flow_count": summary["filtered_flow_count"],
        "selected_flow_count": summary["selected_flow_count"],
        "device_count": summary["device_count"],
    }
    print(json.dumps(printable_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
