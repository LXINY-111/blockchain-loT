import argparse
import csv
import heapq
import io
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from prepare_iot_dataset import (
    CONTROL_PROTOCOLS,
    VALID_FLOW_YEARS,
    DeviceInfo,
    MoteLocation,
    assign_device_topology,
    blockemulator_tx_row,
    clean_cell,
    endpoint_to_eth_address,
    euclidean_distance,
    flow_value,
    format_float,
    format_limit_suffix,
    is_multicast_or_broadcast_ip,
    is_private_ip,
    load_connectivity,
    load_mote_locs,
    lookup_link_quality,
    normalize_mac,
    normalized_protocol,
    number_to_int,
    parse_device_from_flow_name,
    parse_time_key,
    protocol_tokens,
    stable_int,
    usable_endpoint_value,
    write_topology,
)


ORIGINAL_SIDECAR_FIELDS = [
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

MULTI_ANCHOR_FIELDS = ORIGINAL_SIDECAR_FIELDS + [
    "primary_device_ip",
    "flow_direction",
    "relation_type",
    "peer_endpoint",
    "peer_anchor_key",
    "peer_anchor_address",
    "peer_anchor_type",
    "anchor_set_id",
    "anchor_count",
    "anchor_addresses",
    "anchor_types",
    "anchor_roles",
    "anchor_weights",
    "anchor_mote_ids",
    "anchor_distances",
    "anchor_link_qualities",
    "is_control_protocol",
    "is_multicast_or_broadcast",
]

ANCHOR_REGISTRY_FIELDS = [
    "anchor_address",
    "anchor_key",
    "anchor_type",
    "label",
    "device_label",
    "device_mac",
    "endpoint",
    "primary_ip",
    "mapped_mote_id",
    "mapped_x",
    "mapped_y",
]

STATE_ANCHOR_EDGE_FIELDS = [
    "to_address",
    "state_object_key",
    "anchor_address",
    "anchor_key",
    "anchor_type",
    "anchor_role",
    "flow_count",
    "total_payload",
    "total_packets",
    "total_weight",
    "mean_weight",
    "first_tx_index",
    "last_tx_index",
]


@dataclass(frozen=True)
class Anchor:
    key: str
    address: str
    anchor_type: str
    label: str
    endpoint: str
    device_label: str = ""
    device_mac: str = ""
    primary_ip: str = ""


@dataclass
class TempReader:
    index: int
    path: Path
    handle: object
    reader: csv.DictReader


def build_iot_multi_anchor_dataset(
    flows_zip_path: Path,
    mote_locs_path: Path,
    connectivity_path: Path,
    output_dir: Path,
    limit: int = 0,
    tx_batch_size: int = 1000,
) -> Dict[str, object]:
    flows_zip_path = Path(flows_zip_path)
    mote_locs_path = Path(mote_locs_path)
    connectivity_path = Path(connectivity_path)
    output_dir = Path(output_dir)

    validate_inputs(flows_zip_path, mote_locs_path, connectivity_path, limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    mote_locs = load_mote_locs(mote_locs_path)
    connectivity = load_connectivity(connectivity_path)
    temp_dir = output_dir / (
        "_tmp_iot_multi_anchor_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    )
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        with zipfile.ZipFile(flows_zip_path, "r") as archive:
            flow_entries = sorted(
                [entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")],
                key=lambda entry: entry.filename,
            )
            device_infos = {
                entry.filename: parse_device_from_flow_name(entry.filename)
                for entry in flow_entries
            }
            device_topology = assign_device_topology(device_infos.values(), mote_locs)

            temp_paths, scan_stats = write_sorted_temp_files(
                archive=archive,
                flow_entries=flow_entries,
                temp_dir=temp_dir,
            )

        primary_ip_by_flow, ip_to_flow_files, primary_ip_stats = infer_primary_ips(
            scan_stats["local_ip_counts_by_file"]
        )
        unique_ip_to_device = build_unique_ip_device_map(
            ip_to_flow_files=ip_to_flow_files,
            device_infos=device_infos,
        )

        suffix = format_limit_suffix(limit)
        output_suffix = f"multi_anchor_{suffix}"
        tx_path = output_dir / f"selectedTxs_iot_{output_suffix}.csv"
        sidecar_path = output_dir / f"iot_flow_sidecar_{output_suffix}.csv"
        topology_path = output_dir / f"iot_device_topology_{output_suffix}.csv"
        anchor_registry_path = output_dir / f"iot_anchor_registry_{output_suffix}.csv"
        state_anchor_edges_path = output_dir / f"iot_state_anchor_edges_{output_suffix}.csv"
        summary_path = output_dir / f"iot_dataset_{output_suffix}_summary.json"

        result = write_outputs_from_temp(
            temp_paths=temp_paths,
            tx_path=tx_path,
            sidecar_path=sidecar_path,
            anchor_registry_path=anchor_registry_path,
            state_anchor_edges_path=state_anchor_edges_path,
            device_infos=device_infos,
            device_topology=device_topology,
            primary_ip_by_flow=primary_ip_by_flow,
            unique_ip_to_device=unique_ip_to_device,
            mote_locs=mote_locs,
            connectivity=connectivity,
            limit=limit,
            tx_batch_size=tx_batch_size,
        )
        write_topology(topology_path, device_infos.values(), device_topology)

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_variant": "multi_anchor_full" if limit == 0 else f"multi_anchor_{suffix}",
            "input_files": {
                "flows_zip": str(flows_zip_path),
                "mote_locs": str(mote_locs_path),
                "connectivity": str(connectivity_path),
            },
            "output_files": {
                "transactions": str(tx_path),
                "sidecar": str(sidecar_path),
                "topology": str(topology_path),
                "anchor_registry": str(anchor_registry_path),
                "state_anchor_edges": str(state_anchor_edges_path),
                "summary": str(summary_path),
            },
            "limit": limit,
            "tx_batch_size": tx_batch_size,
            "flow_file_count": len(device_infos),
            "device_count": len(device_infos),
            "total_flow_count": scan_stats["total_flow_count"],
            "invalid_time_flow_count": scan_stats["invalid_time_flow_count"],
            "missing_endpoint_flow_count": scan_stats["missing_endpoint_flow_count"],
            "selected_flow_count": result["selected_flow_count"],
            "selected_state_object_count": result["selected_state_object_count"],
            "selected_from_address_count": result["selected_from_address_count"],
            "selected_to_address_count": result["selected_to_address_count"],
            "anchor_registry_count": result["anchor_registry_count"],
            "state_anchor_edge_count": result["state_anchor_edge_count"],
            "multi_anchor_row_count": result["multi_anchor_row_count"],
            "multi_anchor_state_object_count": result["multi_anchor_state_object_count"],
            "anchor_count_distribution": dict(sorted(result["anchor_count_distribution"].items())),
            "anchor_type_distribution_rows": dict(
                sorted(result["anchor_type_distribution_rows"].items())
            ),
            "relation_type_distribution": dict(
                sorted(result["relation_type_distribution"].items())
            ),
            "flow_direction_distribution": dict(
                sorted(result["flow_direction_distribution"].items())
            ),
            "protocol_distribution_seen_valid_time": dict(
                sorted(scan_stats["protocol_distribution_seen_valid_time"].items())
            ),
            "protocol_distribution_selected": dict(
                sorted(result["protocol_distribution_selected"].items())
            ),
            "selected_flow_count_by_device": dict(
                sorted(result["selected_flow_count_by_device"].items())
            ),
            "primary_ip_inference": primary_ip_stats,
            "address_grain": {
                "from_address": "hash(device_label, device_mac)",
                "to_address": "hash(device_label, peer_anchor_identity, dstPort/service, protocol)",
                "state_object_key": "device_label + peer anchor identity + dstPort/service + protocol",
                "multi_anchor_edges": "state object account -> {owner device anchor, observed peer/service anchors}",
            },
            "filter_rules": {
                "kept_protocols": "all protocols with valid timestamps and at least one usable endpoint",
                "valid_time_years": sorted(VALID_FLOW_YEARS),
                "missing_endpoint_policy": "drop only rows without any usable src/dst IP or MAC",
                "control_and_discovery_policy": (
                    "keep ARP/DHCP/ICMP/SSDP/mDNS/etc. as IoT control/discovery flows; "
                    "multicast/broadcast endpoints become service-group anchors"
                ),
            },
            "topology_augmentation": {
                "method": (
                    "UNSW IoT flow devices are deterministically mapped to Intel Berkeley mote "
                    "coordinates; non-device anchors are deterministically mapped to motes by hash "
                    "for distance/link-quality features."
                ),
                "mote_count": len(mote_locs),
                "connectivity_edge_count": len(connectivity),
            },
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        return summary
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def validate_inputs(
    flows_zip_path: Path, mote_locs_path: Path, connectivity_path: Path, limit: int
) -> None:
    for path in (flows_zip_path, mote_locs_path, connectivity_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if limit < 0:
        raise ValueError("limit must be zero for all valid flows or a positive row cap")


def write_sorted_temp_files(
    archive: zipfile.ZipFile,
    flow_entries: Iterable[zipfile.ZipInfo],
    temp_dir: Path,
) -> Tuple[List[Path], Dict[str, object]]:
    temp_paths: List[Path] = []
    total_flow_count = 0
    invalid_time_flow_count = 0
    missing_endpoint_flow_count = 0
    protocol_seen_valid_time: Counter = Counter()
    local_ip_counts_by_file: Dict[str, Counter] = {}

    for file_index, entry in enumerate(flow_entries):
        print(f"[multi-anchor] sorting {entry.filename}", file=sys.stderr, flush=True)
        rows: List[Tuple[float, int, Dict[str, str]]] = []
        local_ip_counts: Counter = Counter()

        with archive.open(entry, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text)
            fieldnames = list(reader.fieldnames or [])
            for local_sequence, row in enumerate(reader):
                total_flow_count += 1
                time_key = parse_time_key(row.get("time", ""))
                if time_key is None:
                    invalid_time_flow_count += 1
                    continue

                protocol_seen_valid_time[normalized_protocol(row)] += 1
                count_local_ips(row, local_ip_counts)
                if not has_any_endpoint(row):
                    missing_endpoint_flow_count += 1
                    continue

                rows.append((time_key, local_sequence, dict(row)))

        rows.sort(key=lambda item: (item[0], item[1]))
        local_ip_counts_by_file[entry.filename] = local_ip_counts

        temp_path = temp_dir / f"flow_{file_index:03d}.csv"
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["__time_key", "__local_sequence", "__flow_file"] + fieldnames,
            )
            writer.writeheader()
            for time_key, local_sequence, row in rows:
                out = {
                    "__time_key": format_float(time_key),
                    "__local_sequence": str(local_sequence),
                    "__flow_file": entry.filename,
                }
                out.update(row)
                writer.writerow(out)
        temp_paths.append(temp_path)
        print(
            f"[multi-anchor] temp {temp_path.name} rows={len(rows)}",
            file=sys.stderr,
            flush=True,
        )

    return temp_paths, {
        "total_flow_count": total_flow_count,
        "invalid_time_flow_count": invalid_time_flow_count,
        "missing_endpoint_flow_count": missing_endpoint_flow_count,
        "protocol_distribution_seen_valid_time": protocol_seen_valid_time,
        "local_ip_counts_by_file": local_ip_counts_by_file,
    }


def count_local_ips(row: Mapping[str, str], counts: Counter) -> None:
    for key in ("srcIp", "dstIp"):
        value = clean_cell(row.get(key, ""))
        if is_local_device_ip(value):
            counts[value] += 1


def has_any_endpoint(row: Mapping[str, str]) -> bool:
    for side in ("src", "dst"):
        if usable_endpoint_value(row.get(f"{side}Ip", "")):
            return True
        if usable_endpoint_value(normalize_mac(row.get(f"{side}Mac", ""))):
            return True
    return False


def infer_primary_ips(
    local_ip_counts_by_file: Mapping[str, Counter],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, object]]:
    primary_ip_by_flow: Dict[str, str] = {}
    ip_to_flow_files: Dict[str, List[str]] = defaultdict(list)
    confidence_by_flow: Dict[str, Dict[str, object]] = {}

    for flow_file, counter in sorted(local_ip_counts_by_file.items()):
        if not counter:
            primary_ip_by_flow[flow_file] = ""
            confidence_by_flow[flow_file] = {
                "primary_ip": "",
                "primary_ip_count": 0,
                "candidate_ip_count": 0,
            }
            continue

        primary_ip, primary_count = counter.most_common(1)[0]
        primary_ip_by_flow[flow_file] = primary_ip
        ip_to_flow_files[primary_ip].append(flow_file)
        confidence_by_flow[flow_file] = {
            "primary_ip": primary_ip,
            "primary_ip_count": primary_count,
            "candidate_ip_count": len(counter),
        }

    duplicate_primary_ips = {
        ip: files for ip, files in sorted(ip_to_flow_files.items()) if len(files) > 1
    }
    stats = {
        "flow_file_count_with_primary_ip": sum(1 for value in primary_ip_by_flow.values() if value),
        "unique_primary_ip_count": len(ip_to_flow_files),
        "duplicate_primary_ips": duplicate_primary_ips,
        "per_flow_file": confidence_by_flow,
    }
    return primary_ip_by_flow, ip_to_flow_files, stats


def build_unique_ip_device_map(
    ip_to_flow_files: Mapping[str, List[str]],
    device_infos: Mapping[str, DeviceInfo],
) -> Dict[str, DeviceInfo]:
    unique: Dict[str, DeviceInfo] = {}
    for ip, flow_files in ip_to_flow_files.items():
        if len(flow_files) == 1:
            unique[ip] = device_infos[flow_files[0]]
    return unique


def write_outputs_from_temp(
    temp_paths: Iterable[Path],
    tx_path: Path,
    sidecar_path: Path,
    anchor_registry_path: Path,
    state_anchor_edges_path: Path,
    device_infos: Mapping[str, DeviceInfo],
    device_topology: Mapping[str, MoteLocation],
    primary_ip_by_flow: Mapping[str, str],
    unique_ip_to_device: Mapping[str, DeviceInfo],
    mote_locs: Mapping[int, MoteLocation],
    connectivity: Mapping[Tuple[int, int], float],
    limit: int,
    tx_batch_size: int,
) -> Dict[str, object]:
    anchor_by_key: Dict[str, Anchor] = {}
    anchor_motes: Dict[str, MoteLocation] = {}
    edge_stats: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    selected_state_objects = set()
    multi_anchor_state_objects = set()
    selected_from_addresses = set()
    selected_to_addresses = set()
    anchor_count_distribution: Counter = Counter()
    anchor_type_distribution_rows: Counter = Counter()
    relation_type_distribution: Counter = Counter()
    flow_direction_distribution: Counter = Counter()
    protocol_distribution_selected: Counter = Counter()
    selected_flow_count_by_device: Counter = Counter()
    multi_anchor_row_count = 0

    readers = open_temp_readers(temp_paths)
    heap: List[Tuple[float, int, int, TempReader, Dict[str, str]]] = []
    for temp_reader in readers:
        push_next_temp_row(heap, temp_reader)

    try:
        with tx_path.open("w", encoding="utf-8", newline="") as tx_handle, sidecar_path.open(
            "w", encoding="utf-8", newline=""
        ) as sidecar_handle:
            tx_writer = csv.writer(tx_handle)
            sidecar_writer = csv.DictWriter(sidecar_handle, fieldnames=MULTI_ANCHOR_FIELDS)
            sidecar_writer.writeheader()

            tx_index = 0
            while heap:
                _time_key, _reader_index, _local_sequence, temp_reader, row = heapq.heappop(heap)
                flow_file = clean_cell(row["__flow_file"])
                device = device_infos[flow_file]
                protocol = normalized_protocol(row)
                primary_ip = primary_ip_by_flow.get(flow_file, "")
                owner_anchor = device_anchor(
                    device=device,
                    primary_ip=primary_ip,
                    device_topology=device_topology,
                    anchor_by_key=anchor_by_key,
                    anchor_motes=anchor_motes,
                )
                endpoint_anchors = [
                    anchor
                    for anchor in (
                        endpoint_anchor(
                            row=row,
                            side="src",
                            owner_device=device,
                            owner_anchor=owner_anchor,
                            unique_ip_to_device=unique_ip_to_device,
                            device_topology=device_topology,
                            primary_ip_by_flow=primary_ip_by_flow,
                            mote_locs=mote_locs,
                            anchor_by_key=anchor_by_key,
                            anchor_motes=anchor_motes,
                        ),
                        endpoint_anchor(
                            row=row,
                            side="dst",
                            owner_device=device,
                            owner_anchor=owner_anchor,
                            unique_ip_to_device=unique_ip_to_device,
                            device_topology=device_topology,
                            primary_ip_by_flow=primary_ip_by_flow,
                            mote_locs=mote_locs,
                            anchor_by_key=anchor_by_key,
                            anchor_motes=anchor_motes,
                        ),
                    )
                    if anchor is not None
                ]
                anchors = unique_anchors([owner_anchor] + endpoint_anchors)
                peer_anchors = [anchor for anchor in anchors if anchor.address != owner_anchor.address]
                peer_anchor = choose_peer_anchor(row, primary_ip, peer_anchors)
                state_key = multi_anchor_state_object_key(device, row, peer_anchors, protocol)
                state_address = endpoint_to_eth_address(state_key)
                from_address = owner_anchor.address
                value = flow_value(row)

                tx_writer.writerow(blockemulator_tx_row(from_address, state_address, value))
                tx_batch_id = tx_index // tx_batch_size

                owner_mote = anchor_motes[owner_anchor.key]
                related_anchor = peer_anchor if peer_anchor is not None else owner_anchor
                related_mote = anchor_motes[related_anchor.key]
                distance = euclidean_distance(owner_mote, related_mote)
                link_quality = lookup_link_quality(
                    connectivity, owner_mote.mote_id, related_mote.mote_id, distance
                )
                anchor_weights = normalized_anchor_weights(anchors)
                anchor_set_id = endpoint_to_eth_address(
                    "iot_anchor_set:" + "|".join(sorted(anchor.address for anchor in anchors))
                )
                anchor_distances, anchor_link_qualities = anchor_cost_lists(
                    anchors, owner_mote, anchor_motes, connectivity
                )
                relation_type = classify_relation(peer_anchors)
                flow_direction = classify_direction(row, primary_ip)
                has_control = bool(protocol_tokens(row) & CONTROL_PROTOCOLS)
                has_multicast = is_multicast_or_broadcast_ip(row.get("srcIp", "")) or (
                    is_multicast_or_broadcast_ip(row.get("dstIp", ""))
                )

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
                        "to_address": state_address,
                        "mapped_mote_id": owner_mote.mote_id,
                        "mapped_x": format_float(owner_mote.x),
                        "mapped_y": format_float(owner_mote.y),
                        "related_mote_id": related_mote.mote_id,
                        "related_x": format_float(related_mote.x),
                        "related_y": format_float(related_mote.y),
                        "distance": format_float(distance),
                        "link_quality": format_float(link_quality),
                        "tx_batch_id": tx_batch_id,
                        "primary_device_ip": primary_ip,
                        "flow_direction": flow_direction,
                        "relation_type": relation_type,
                        "peer_endpoint": peer_anchor.endpoint if peer_anchor else "",
                        "peer_anchor_key": peer_anchor.key if peer_anchor else "",
                        "peer_anchor_address": peer_anchor.address if peer_anchor else "",
                        "peer_anchor_type": peer_anchor.anchor_type if peer_anchor else "",
                        "anchor_set_id": anchor_set_id,
                        "anchor_count": len(anchors),
                        "anchor_addresses": join_values(anchor.address for anchor in anchors),
                        "anchor_types": join_values(anchor.anchor_type for anchor in anchors),
                        "anchor_roles": join_values(
                            "owner" if anchor.address == owner_anchor.address else "peer"
                            for anchor in anchors
                        ),
                        "anchor_weights": join_values(format_float(weight) for weight in anchor_weights),
                        "anchor_mote_ids": join_values(str(anchor_motes[anchor.key].mote_id) for anchor in anchors),
                        "anchor_distances": join_values(anchor_distances),
                        "anchor_link_qualities": join_values(anchor_link_qualities),
                        "is_control_protocol": int(has_control),
                        "is_multicast_or_broadcast": int(has_multicast),
                    }
                )

                update_edge_stats(
                    edge_stats=edge_stats,
                    tx_index=tx_index,
                    state_key=state_key,
                    state_address=state_address,
                    anchors=anchors,
                    owner_address=owner_anchor.address,
                    weights=anchor_weights,
                    row=row,
                )

                selected_state_objects.add(state_key)
                if len(anchors) > 1:
                    multi_anchor_row_count += 1
                    multi_anchor_state_objects.add(state_key)
                selected_from_addresses.add(from_address)
                selected_to_addresses.add(state_address)
                anchor_count_distribution[len(anchors)] += 1
                relation_type_distribution[relation_type] += 1
                flow_direction_distribution[flow_direction] += 1
                protocol_distribution_selected[protocol] += 1
                selected_flow_count_by_device[device.device_label] += 1
                for anchor in anchors:
                    anchor_type_distribution_rows[anchor.anchor_type] += 1

                tx_index += 1
                if tx_index % 250000 == 0:
                    print(
                        f"[multi-anchor] wrote rows={tx_index}",
                        file=sys.stderr,
                        flush=True,
                    )
                if limit > 0 and tx_index >= limit:
                    break
                push_next_temp_row(heap, temp_reader)

        write_anchor_registry(anchor_registry_path, anchor_by_key, anchor_motes)
        write_state_anchor_edges(state_anchor_edges_path, edge_stats)

        return {
            "selected_flow_count": tx_index,
            "selected_state_object_count": len(selected_state_objects),
            "selected_from_address_count": len(selected_from_addresses),
            "selected_to_address_count": len(selected_to_addresses),
            "anchor_registry_count": len(anchor_by_key),
            "state_anchor_edge_count": len(edge_stats),
            "multi_anchor_row_count": multi_anchor_row_count,
            "multi_anchor_state_object_count": len(multi_anchor_state_objects),
            "anchor_count_distribution": anchor_count_distribution,
            "anchor_type_distribution_rows": anchor_type_distribution_rows,
            "relation_type_distribution": relation_type_distribution,
            "flow_direction_distribution": flow_direction_distribution,
            "protocol_distribution_selected": protocol_distribution_selected,
            "selected_flow_count_by_device": selected_flow_count_by_device,
        }
    finally:
        for temp_reader in readers:
            temp_reader.handle.close()


def open_temp_readers(temp_paths: Iterable[Path]) -> List[TempReader]:
    readers: List[TempReader] = []
    for index, path in enumerate(temp_paths):
        handle = path.open("r", encoding="utf-8", newline="")
        readers.append(TempReader(index=index, path=path, handle=handle, reader=csv.DictReader(handle)))
    return readers


def push_next_temp_row(
    heap: List[Tuple[float, int, int, TempReader, Dict[str, str]]],
    temp_reader: TempReader,
) -> None:
    row = next(temp_reader.reader, None)
    if row is None:
        return
    time_key = float(row["__time_key"])
    local_sequence = int(row["__local_sequence"])
    heapq.heappush(heap, (time_key, temp_reader.index, local_sequence, temp_reader, row))


def device_anchor(
    device: DeviceInfo,
    primary_ip: str,
    device_topology: Mapping[str, MoteLocation],
    anchor_by_key: Dict[str, Anchor],
    anchor_motes: Dict[str, MoteLocation],
) -> Anchor:
    key = device_anchor_key(device)
    if key not in anchor_by_key:
        anchor_by_key[key] = Anchor(
            key=key,
            address=endpoint_to_eth_address(key),
            anchor_type="device",
            label=device.device_label,
            endpoint=primary_ip,
            device_label=device.device_label,
            device_mac=device.device_mac,
            primary_ip=primary_ip,
        )
        anchor_motes[key] = device_topology[device.flow_file]
    return anchor_by_key[key]


def endpoint_anchor(
    row: Mapping[str, str],
    side: str,
    owner_device: DeviceInfo,
    owner_anchor: Anchor,
    unique_ip_to_device: Mapping[str, DeviceInfo],
    device_topology: Mapping[str, MoteLocation],
    primary_ip_by_flow: Mapping[str, str],
    mote_locs: Mapping[int, MoteLocation],
    anchor_by_key: Dict[str, Anchor],
    anchor_motes: Dict[str, MoteLocation],
) -> Optional[Anchor]:
    ip_value = clean_cell(row.get(f"{side}Ip", "")).lower()
    if usable_endpoint_value(ip_value):
        if ip_value == owner_anchor.primary_ip:
            return owner_anchor
        if ip_value in unique_ip_to_device:
            device = unique_ip_to_device[ip_value]
            return device_anchor(
                device=device,
                primary_ip=primary_ip_by_flow.get(device.flow_file, ip_value),
                device_topology=device_topology,
                anchor_by_key=anchor_by_key,
                anchor_motes=anchor_motes,
            )
        key, anchor_type, label = endpoint_anchor_identity(row, ip_value)
        return register_non_device_anchor(
            key=key,
            anchor_type=anchor_type,
            label=label,
            endpoint=ip_value,
            mote_locs=mote_locs,
            anchor_by_key=anchor_by_key,
            anchor_motes=anchor_motes,
        )

    mac_value = normalize_mac(row.get(f"{side}Mac", ""))
    if usable_endpoint_value(mac_value):
        key = f"iot_mac_endpoint:{mac_value}"
        return register_non_device_anchor(
            key=key,
            anchor_type="mac_endpoint",
            label=mac_value,
            endpoint=mac_value,
            mote_locs=mote_locs,
            anchor_by_key=anchor_by_key,
            anchor_motes=anchor_motes,
        )
    return None


def endpoint_anchor_identity(row: Mapping[str, str], ip_value: str) -> Tuple[str, str, str]:
    protocol = normalized_protocol(row)
    service = clean_cell(row.get("dstPort", "")) or "any"
    if ip_value == "192.168.1.1":
        return "iot_gateway:192.168.1.1", "gateway", "gateway:192.168.1.1"
    if is_multicast_or_broadcast_ip(ip_value):
        key = f"iot_service_group:{ip_value}|service:{service}|proto:{protocol}"
        return key, "service_group", f"group:{ip_value}"
    if is_local_device_ip(ip_value):
        return f"iot_local_endpoint:{ip_value}", "local_endpoint", f"local:{ip_value}"
    if is_private_ip(ip_value):
        return f"iot_private_endpoint:{ip_value}", "private_endpoint", f"private:{ip_value}"
    return f"iot_cloud_endpoint:{ip_value}", "cloud_endpoint", f"cloud:{ip_value}"


def register_non_device_anchor(
    key: str,
    anchor_type: str,
    label: str,
    endpoint: str,
    mote_locs: Mapping[int, MoteLocation],
    anchor_by_key: Dict[str, Anchor],
    anchor_motes: Dict[str, MoteLocation],
) -> Anchor:
    if key not in anchor_by_key:
        anchor_by_key[key] = Anchor(
            key=key,
            address=endpoint_to_eth_address(key),
            anchor_type=anchor_type,
            label=label,
            endpoint=endpoint,
        )
        anchor_motes[key] = deterministic_mote_for_anchor(key, mote_locs)
    return anchor_by_key[key]


def deterministic_mote_for_anchor(key: str, motes: Mapping[int, MoteLocation]) -> MoteLocation:
    mote_ids = list(motes.keys())
    return motes[mote_ids[stable_int(key) % len(mote_ids)]]


def unique_anchors(anchors: Iterable[Anchor]) -> List[Anchor]:
    out: List[Anchor] = []
    seen = set()
    for anchor in anchors:
        if anchor.address in seen:
            continue
        seen.add(anchor.address)
        out.append(anchor)
    return out


def choose_peer_anchor(
    row: Mapping[str, str], primary_ip: str, peer_anchors: List[Anchor]
) -> Optional[Anchor]:
    if not peer_anchors:
        return None
    src_ip = clean_cell(row.get("srcIp", "")).lower()
    dst_ip = clean_cell(row.get("dstIp", "")).lower()
    if src_ip == primary_ip and dst_ip:
        for anchor in peer_anchors:
            if anchor.endpoint == dst_ip:
                return anchor
    if dst_ip == primary_ip and src_ip:
        for anchor in peer_anchors:
            if anchor.endpoint == src_ip:
                return anchor
    return peer_anchors[0]


def multi_anchor_state_object_key(
    device: DeviceInfo,
    row: Mapping[str, str],
    peer_anchors: List[Anchor],
    protocol: str,
) -> str:
    label = clean_label(device.device_label)
    service = clean_cell(row.get("dstPort", "")) or "any"
    if peer_anchors:
        peer_part = "+".join(sorted(anchor_state_label(anchor) for anchor in peer_anchors))
    else:
        peer_part = "owner"
    return f"iot_state_multi:device:{label}|peer:{peer_part}|service:{service}|proto:{protocol}"


def anchor_state_label(anchor: Anchor) -> str:
    if anchor.anchor_type == "device":
        return "device:" + clean_label(anchor.device_label)
    if anchor.anchor_type == "gateway":
        return "gateway:" + anchor.endpoint
    if anchor.anchor_type == "service_group":
        return "group:" + anchor.endpoint
    if anchor.anchor_type == "cloud_endpoint":
        return "cloud:" + anchor.endpoint
    if anchor.anchor_type == "local_endpoint":
        return "local:" + anchor.endpoint
    if anchor.anchor_type == "private_endpoint":
        return "private:" + anchor.endpoint
    return anchor.anchor_type + ":" + clean_label(anchor.endpoint)


def normalized_anchor_weights(anchors: List[Anchor]) -> List[float]:
    if not anchors:
        return []
    weight = 1.0 / float(len(anchors))
    return [weight for _anchor in anchors]


def anchor_cost_lists(
    anchors: List[Anchor],
    owner_mote: MoteLocation,
    anchor_motes: Mapping[str, MoteLocation],
    connectivity: Mapping[Tuple[int, int], float],
) -> Tuple[List[str], List[str]]:
    distances: List[str] = []
    link_qualities: List[str] = []
    for anchor in anchors:
        mote = anchor_motes[anchor.key]
        distance = euclidean_distance(owner_mote, mote)
        link_quality = lookup_link_quality(connectivity, owner_mote.mote_id, mote.mote_id, distance)
        distances.append(format_float(distance))
        link_qualities.append(format_float(link_quality))
    return distances, link_qualities


def classify_relation(peer_anchors: List[Anchor]) -> str:
    if not peer_anchors:
        return "owner_only"
    peer_types = {anchor.anchor_type for anchor in peer_anchors}
    if len(peer_types) > 1:
        return "device_to_multi_anchor"
    peer_type = next(iter(peer_types))
    mapping = {
        "device": "device_to_device",
        "gateway": "device_to_gateway",
        "service_group": "device_to_service_group",
        "cloud_endpoint": "device_to_cloud",
        "local_endpoint": "device_to_local_endpoint",
        "private_endpoint": "device_to_private_endpoint",
        "mac_endpoint": "device_to_mac_endpoint",
    }
    return mapping.get(peer_type, "device_to_other")


def classify_direction(row: Mapping[str, str], primary_ip: str) -> str:
    src_ip = clean_cell(row.get("srcIp", "")).lower()
    dst_ip = clean_cell(row.get("dstIp", "")).lower()
    if not primary_ip:
        return "unknown_primary_ip"
    if src_ip == primary_ip and dst_ip == primary_ip:
        return "self"
    if src_ip == primary_ip:
        return "outbound"
    if dst_ip == primary_ip:
        return "inbound"
    return "owner_not_in_ip_endpoints"


def update_edge_stats(
    edge_stats: Dict[Tuple[str, str, str], Dict[str, object]],
    tx_index: int,
    state_key: str,
    state_address: str,
    anchors: List[Anchor],
    owner_address: str,
    weights: List[float],
    row: Mapping[str, str],
) -> None:
    payload = number_to_int(row.get("srcPayloadSize", "")) + number_to_int(
        row.get("dstPayloadSize", "")
    )
    packets = number_to_int(row.get("srcNumPackets", "")) + number_to_int(
        row.get("dstNumPackets", "")
    )
    for anchor, weight in zip(anchors, weights):
        role = "owner" if anchor.address == owner_address else "peer"
        key = (state_address, anchor.address, role)
        if key not in edge_stats:
            edge_stats[key] = {
                "to_address": state_address,
                "state_object_key": state_key,
                "anchor_address": anchor.address,
                "anchor_key": anchor.key,
                "anchor_type": anchor.anchor_type,
                "anchor_role": role,
                "flow_count": 0,
                "total_payload": 0,
                "total_packets": 0,
                "total_weight": 0.0,
                "first_tx_index": tx_index,
                "last_tx_index": tx_index,
            }
        stat = edge_stats[key]
        stat["flow_count"] = int(stat["flow_count"]) + 1
        stat["total_payload"] = int(stat["total_payload"]) + payload
        stat["total_packets"] = int(stat["total_packets"]) + packets
        stat["total_weight"] = float(stat["total_weight"]) + weight
        stat["last_tx_index"] = tx_index


def write_anchor_registry(
    path: Path,
    anchor_by_key: Mapping[str, Anchor],
    anchor_motes: Mapping[str, MoteLocation],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANCHOR_REGISTRY_FIELDS)
        writer.writeheader()
        for key in sorted(anchor_by_key):
            anchor = anchor_by_key[key]
            mote = anchor_motes[key]
            writer.writerow(
                {
                    "anchor_address": anchor.address,
                    "anchor_key": anchor.key,
                    "anchor_type": anchor.anchor_type,
                    "label": anchor.label,
                    "device_label": anchor.device_label,
                    "device_mac": anchor.device_mac,
                    "endpoint": anchor.endpoint,
                    "primary_ip": anchor.primary_ip,
                    "mapped_mote_id": mote.mote_id,
                    "mapped_x": format_float(mote.x),
                    "mapped_y": format_float(mote.y),
                }
            )


def write_state_anchor_edges(
    path: Path,
    edge_stats: Mapping[Tuple[str, str, str], Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_ANCHOR_EDGE_FIELDS)
        writer.writeheader()
        for key in sorted(edge_stats):
            stat = dict(edge_stats[key])
            flow_count = int(stat["flow_count"])
            stat["mean_weight"] = format_float(float(stat["total_weight"]) / max(1, flow_count))
            stat["total_weight"] = format_float(float(stat["total_weight"]))
            writer.writerow(stat)


def device_anchor_key(device: DeviceInfo) -> str:
    label = clean_label(device.device_label) or "unknown_device"
    mac = clean_cell(device.device_mac).lower() or "unknown_mac"
    return f"iot_device:{label}|mac:{mac}"


def is_local_device_ip(value: str) -> bool:
    text = clean_cell(value)
    if not usable_endpoint_value(text):
        return False
    if text in {"192.168.1.1", "192.168.1.255"}:
        return False
    return text.startswith("192.168.1.")


def clean_label(value: str) -> str:
    text = clean_cell(value).lower()
    return re.sub(r"[^0-9a-z_.:-]+", "_", text).strip("_")


def join_values(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a full multi-anchor IoT dataset for BlockEmulator + SPRING/PPO experiments."
        )
    )
    parser.add_argument("--flows-zip", type=Path, default=Path(r"E:\flows.zip"))
    parser.add_argument("--mote-locs", type=Path, default=Path(r"E:\mote_locs.txt"))
    parser.add_argument("--connectivity", type=Path, default=Path(r"E:\connectivity.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_iot"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tx-batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_iot_multi_anchor_dataset(
        flows_zip_path=args.flows_zip,
        mote_locs_path=args.mote_locs,
        connectivity_path=args.connectivity,
        output_dir=args.output_dir,
        limit=args.limit,
        tx_batch_size=args.tx_batch_size,
    )
    printable = {
        "transactions": summary["output_files"]["transactions"],
        "sidecar": summary["output_files"]["sidecar"],
        "anchor_registry": summary["output_files"]["anchor_registry"],
        "state_anchor_edges": summary["output_files"]["state_anchor_edges"],
        "summary": summary["output_files"]["summary"],
        "selected_flow_count": summary["selected_flow_count"],
        "selected_state_object_count": summary["selected_state_object_count"],
        "anchor_registry_count": summary["anchor_registry_count"],
        "state_anchor_edge_count": summary["state_anchor_edge_count"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
