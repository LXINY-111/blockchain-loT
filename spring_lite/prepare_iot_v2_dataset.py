"""真实交易骨架 + 可追溯物联网场景。只生成数据，不导入或修改训练/奖励代码。

账户与场景映射是仿真假设：27 类设备画像、54 个逻辑位置均可复用；
原始 flow 不是对应链上交易的实测流量，flow 时间不是链上时间。
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "real_accounts_iot_scene_v2_1"
ROOT = Path(__file__).resolve().parents[1]
CONTROL = {"arp", "dhcp", "dhcpv6", "icmp", "icmpv6", "igmp", "llmnr", "mdns", "ssdp"}
FLOW_NUMERIC = ("srcNumPackets", "dstNumPackets", "srcPayloadSize", "dstPayloadSize")
TEMPLATE_FIELDS = ["template_id", "profile_id", "source_file", "source_row", "time",
                   "protocol", "srcIp", "dstIp", "srcPort", "dstPort", *FLOW_NUMERIC,
                   "flowDuration", "is_control_protocol"]
ACCOUNT_FIELDS = ["account_address", "profile_id", "device_label", "actor_type",
                  "site_id", "x_m", "y_m", "first_tx_index"]
SCENE_FIELDS = ["tx_index", "source_row", "from_address", "to_address",
                "from_profile_id", "to_profile_id", "from_site_id", "to_site_id",
                "from_actor_type", "to_actor_type", "template_id", "protocol",
                *FLOW_NUMERIC, "flowDuration", "is_control_protocol",
                "distance_m", "link_quality_forward", "link_quality_reverse",
                "link_quality_rule"]
PROFILE_FIELDS = ["profile_id", "device_label", "device_mac", "source_file",
                  "raw_rows", "eligible_rows", "sample_rows", "rejected_rows",
                  "time_min", "time_max", "duration_max_raw"]
LIBRARY_FILES = ["flow_templates.csv", "device_profiles.csv", "sites.csv",
                 "site_links.csv", "scene_library.json"]
SOURCES = {
    "unsw": "https://iotanalytics.unsw.edu.au/unsw-iotraffic.html",
    "dryad": "https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b",
    "intel": "https://db.csail.mit.edu/labdata/labdata.html",
}


def stable_int(seed, purpose, *parts):
    # 使用稳定摘要，不使用 Python 每次进程可能变化的 hash()。
    message = json.dumps([VERSION, int(seed), purpose, *parts],
                         ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:16], "big")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_topology(directory):
    locations, links, ignored = {}, {}, Counter()
    for number, line in enumerate((directory / "mote_locs.txt").read_text().splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 3:
            raise ValueError(f"Invalid location row {number}")
        key, x, y = int(values[0]), float(values[1]), float(values[2])
        if key in locations or not all(map(math.isfinite, (x, y))):
            raise ValueError(f"Duplicate/nonfinite location {key}")
        locations[key] = (x, y)
    for number, line in enumerate((directory / "connectivity.txt").read_text().splitlines(), 1):
        values = line.split()
        if not values:
            continue
        # 原文件末尾可能含非表格控制符；仅允许纯空白/ASCII EOF 行。
        if len(values) != 3:
            if line.strip() == "\x1a":
                ignored["ascii_eof_lines"] += 1
                continue
            # 本地矩阵含无坐标的 0 号节点空质量行。仅排除这种无位置端点，
            # 不能把真实 1..54 节点的缺失质量当作零或反向链路补齐。
            if len(values) == 2 and all(v.isdigit() for v in values) and (
                int(values[0]) not in locations or int(values[1]) not in locations
            ):
                ignored["incomplete_links_without_coordinates"] += 1
                continue
            raise ValueError(f"Invalid connectivity row {number}: {line!r}")
        a, b, quality = int(values[0]), int(values[1]), float(values[2])
        if not math.isfinite(quality) or not 0 <= quality <= 1 or (a, b) in links:
            raise ValueError(f"Duplicate/invalid directed link {a}->{b}")
        if a not in locations or b not in locations:
            ignored["links_without_coordinates"] += 1
            continue
        links[a, b] = quality
    if not locations:
        raise ValueError("No deployment positions")
    missing = [(a, b) for a in locations for b in locations if (a, b) not in links]
    if missing:
        raise ValueError(f"Missing directed links; no reverse/distance imputation: {missing[:5]}")
    return locations, links, dict(ignored)


def numeric_flow(row):
    """校验原始统计量；不把缺失值变为零，不改写合法的零流量/零时长。"""
    for field in FLOW_NUMERIC:
        if not str(row.get(field, "")).strip().isdigit():
            return None, "invalid_" + field
    try:
        duration = float(row["flowDuration"])
        timestamp = datetime.fromisoformat(row["time"].strip())
    except (KeyError, ValueError):
        return None, "invalid_duration_or_time"
    if not math.isfinite(duration) or duration < 0:
        return None, "invalid_duration"
    if timestamp.year not in (2016, 2017):
        return None, "unexpected_source_year"
    return (timestamp.isoformat(sep=" "), duration), None


def create_library(source_dir, output, seed, cap):
    locations, links, ignored = read_topology(source_dir)
    write_csv(output / "sites.csv", ["site_id", "x_m", "y_m"],
              [dict(site_id=k, x_m=v[0], y_m=v[1]) for k, v in sorted(locations.items())])
    # 保留有向原始概率，包括真实的零。共同位置的有效链路另以规则显式标记。
    write_csv(output / "site_links.csv",
              ["from_site_id", "to_site_id", "measured_quality", "effective_quality", "rule"],
              [dict(from_site_id=a, to_site_id=b, measured_quality=q,
                    effective_quality=1.0 if a == b else q,
                    rule="co_located_assumption" if a == b else "measured_directed")
               for (a, b), q in sorted(links.items())])
    profiles, all_samples, rejection_totals = [], [], Counter()
    with zipfile.ZipFile(source_dir / "flows.zip") as archive:
        entries = sorted((e for e in archive.infolist() if e.filename.endswith(".csv")),
                         key=lambda e: e.filename)
        for number, entry in enumerate(entries):
            match = re.fullmatch(r"(.+)_([0-9a-fA-F]{12})_flows.csv", Path(entry.filename).name)
            if not match:
                raise ValueError(f"Unrecognized device filename: {entry.filename}")
            pid = f"profile_{number:02d}"
            rng = random.Random(stable_int(seed, "reservoir", entry.filename))
            pool, raw_count, eligible, rejected = [], 0, 0, Counter()
            time_min, time_max, duration_max = None, None, 0.0
            with archive.open(entry) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                required = set(FLOW_NUMERIC) | {"time", "flowDuration", "protocol"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing flow columns: {entry.filename}")
                for source_row, row in enumerate(reader, 1):
                    raw_count += 1
                    parsed, error = numeric_flow(row)
                    if error:
                        rejected[error] += 1
                        continue
                    timestamp, duration = parsed
                    eligible += 1
                    time_min = timestamp if time_min is None else min(time_min, timestamp)
                    time_max = timestamp if time_max is None else max(time_max, timestamp)
                    duration_max = max(duration_max, duration)
                    # 每种设备独立蓄水池抽样；固定容量保证更大输入也不会全量驻留内存。
                    slot = eligible - 1 if eligible <= cap else rng.randrange(eligible)
                    if slot >= cap:
                        continue
                    protocol = row["protocol"].strip().lower() or "none"
                    sample = {key: row.get(key, "").strip() for key in TEMPLATE_FIELDS}
                    sample.update(template_id=f"{pid}:{source_row}", profile_id=pid,
                                  source_file=entry.filename, source_row=source_row,
                                  time=timestamp, protocol=protocol,
                                  is_control_protocol=int(protocol in CONTROL))
                    if eligible <= cap:
                        pool.append(sample)
                    else:
                        pool[slot] = sample
            if not pool:
                raise ValueError(f"No valid traffic templates for {entry.filename}")
            pool.sort(key=lambda sample: sample["source_row"])
            all_samples.extend(pool)
            mac = match.group(2).lower()
            profiles.append(dict(profile_id=pid, device_label=match.group(1),
                                 device_mac=":".join(mac[i:i+2] for i in range(0, 12, 2)),
                                 source_file=entry.filename, raw_rows=raw_count,
                                 eligible_rows=eligible, sample_rows=len(pool),
                                 rejected_rows=sum(rejected.values()), time_min=time_min,
                                 time_max=time_max, duration_max_raw=duration_max))
            rejection_totals.update(rejected)
            print(f"library {number+1}/{len(entries)}: {match.group(1)} "
                  f"rows={raw_count} eligible={eligible} samples={len(pool)}", flush=True)
    write_csv(output / "flow_templates.csv", TEMPLATE_FIELDS, all_samples)
    write_csv(output / "device_profiles.csv", PROFILE_FIELDS, profiles)
    metadata = dict(version=VERSION, seed=seed, templates_per_profile=cap,
                    source_sha256={name: digest(source_dir / name)
                                   for name in ("flows.zip", "mote_locs.txt", "connectivity.txt")},
                    source_paths={name: str((source_dir / name).resolve())
                                  for name in ("flows.zip", "mote_locs.txt", "connectivity.txt")},
                    source_urls=SOURCES, profiles=len(profiles), sites=len(locations),
                    raw_flows=sum(p["raw_rows"] for p in profiles),
                    eligible_flows=sum(p["eligible_rows"] for p in profiles),
                    sampled_templates=len(all_samples), rejected_flows=dict(rejection_totals),
                    topology_ignored=ignored, duration_unit="source_raw_no_conversion",
                    duration_unit_note="Publisher README labels seconds; local raw values retained. "
                    "Do not label derived rates as bytes/s before extractor/PCAP verification.",
                    mapping_assumptions=dict(account_kind="logical_iot_device",
                        profile_prior="uniform_across_device_profiles",
                        site_prior="uniform_across_coordinate_sites",
                        co_located_quality=1.0, quality_zero="preserved",
                        edge_cloud_service_roles="not_invented"))
    metadata["library_file_sha256"] = {name: digest(output / name) for name in LIBRARY_FILES[:-1]}
    write_json(output / "scene_library.json", metadata)
    return metadata


def reuse_library(source, output, seed, cap):
    metadata = json.loads((source / "scene_library.json").read_text(encoding="utf-8"))
    if (metadata["version"], metadata["seed"], metadata["templates_per_profile"]) != (VERSION, seed, cap):
        raise ValueError("Frozen scene library version/seed/cap must match")
    for name, expected in metadata["library_file_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Changed library file: {name}")
    for name in LIBRARY_FILES:
        shutil.copyfile(source / name, output / name)
    return metadata


def load_library(directory):
    profiles = list(read_csv(directory / "device_profiles.csv"))
    sites = {int(row["site_id"]): (float(row["x_m"]), float(row["y_m"]))
             for row in read_csv(directory / "sites.csv")}
    links = {(int(row["from_site_id"]), int(row["to_site_id"])):
             (float(row["effective_quality"]), row["rule"])
             for row in read_csv(directory / "site_links.csv")}
    templates = {p["profile_id"]: [] for p in profiles}
    for row in read_csv(directory / "flow_templates.csv"):
        templates[row["profile_id"]].append(row)
    return profiles, sites, links, templates


def map_account(address, profiles, sites, seed):
    normalized = address.lower()
    profile = profiles[stable_int(seed, "profile", normalized) % len(profiles)]
    ids = sorted(sites)
    site = ids[stable_int(seed, "site", normalized) % len(ids)]
    return dict(account_address=normalized, profile_id=profile["profile_id"],
                device_label=profile["device_label"], actor_type="device",
                site_id=site, x_m=sites[site][0], y_m=sites[site][1])


def check_chain_row(row, number):
    # 只接受本系统的 18 列无表头格式；坏行报错，不静默删交易或改金额。
    if len(row) != 18:
        raise ValueError(f"Chain row {number}: expected 18 columns, got {len(row)}")
    if row[6:8] != ["0", "0"]:
        raise ValueError(f"Chain row {number}: unsupported flags")
    if any(not re.fullmatch(r"0x[0-9a-f]{40}", row[k]) for k in (3, 4)):
        raise ValueError(f"Chain row {number}: expected lowercase 0x + 40 hex addresses")
    if row[3] == row[4] or not re.fullmatch(r"[0-9]+", row[8]):
        raise ValueError(f"Chain row {number}: self-transfer/invalid integer value")
    return row[3], row[4]


def template_for(row, global_index, account, templates, seed):
    pool = templates[account["profile_id"]]
    # 不读取账户全程频率、未来邻居、模型动作或分片。追加数据不影响既有前缀。
    key = stable_int(seed, "transaction_template", global_index, row)
    return pool[key % len(pool)]


def build(args):
    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refuse to overwrite nonempty dataset directory: {output}")
    if args.templates_per_profile <= 0 or args.index_offset < 0:
        raise ValueError("templates-per-profile > 0 and index-offset >= 0 required")
    if output == args.chain_csv.resolve() or output in args.chain_csv.resolve().parents:
        raise ValueError("Output must not contain the source transaction file")
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status="building", version=VERSION, command=sys.argv)
    write_json(output / "build_status.json", status)
    try:
        metadata = (reuse_library(args.reuse_library.resolve(), output, args.seed,
                                  args.templates_per_profile) if args.reuse_library else
                    create_library(args.iot_source_dir.resolve(), output, args.seed,
                                   args.templates_per_profile))
        profiles, sites, links, templates = load_library(output)
        # 交易文件直接按字节复制，保证数值、空字段、顺序与源文件完全相同。
        transactions = output / "selectedTxs_iot_v2.csv"
        source_sha = digest(args.chain_csv)
        shutil.copyfile(args.chain_csv, transactions)
        if digest(transactions) != source_sha:
            raise RuntimeError("Source changed during transaction copy")
        accounts, template_use, profile_use, site_use, pair_counts = {}, Counter(), Counter(), Counter(), Counter()
        zero_links, co_located, zero_duration, control, rows = 0, 0, 0, 0, 0
        sample_examples = []
        with transactions.open(encoding="utf-8-sig", newline="") as handle, (
            output / "transaction_scene.csv").open("w", encoding="utf-8", newline="") as scene:
            writer = csv.DictWriter(scene, fieldnames=SCENE_FIELDS, lineterminator="\n")
            writer.writeheader()
            for local_index, row in enumerate(csv.reader(handle)):
                a, b = check_chain_row(row, local_index + 1)
                global_index = args.index_offset + local_index
                for address in (a, b):
                    if address not in accounts:
                        accounts[address] = map_account(address, profiles, sites, args.seed)
                        accounts[address]["first_tx_index"] = global_index
                sender, recipient = accounts[a], accounts[b]
                sample = template_for(row, global_index, sender, templates, args.seed)
                u, v = sender["site_id"], recipient["site_id"]
                q_forward, rule = links[u, v]
                q_reverse, _ = links[v, u]
                distance = math.hypot(sender["x_m"] - recipient["x_m"], sender["y_m"] - recipient["y_m"])
                record = dict(tx_index=global_index, source_row=local_index+1,
                    from_address=a, to_address=b, from_profile_id=sender["profile_id"],
                    to_profile_id=recipient["profile_id"], from_site_id=u, to_site_id=v,
                    from_actor_type="device", to_actor_type="device",
                    template_id=sample["template_id"], protocol=sample["protocol"],
                    **{key: sample[key] for key in (*FLOW_NUMERIC, "flowDuration", "is_control_protocol")},
                    distance_m=format(distance, ".12g"),
                    link_quality_forward=format(q_forward, ".17g"),
                    link_quality_reverse=format(q_reverse, ".17g"), link_quality_rule=rule)
                writer.writerow(record)
                rows += 1
                pair_counts[a, b] += 1
                template_use[sample["template_id"]] += 1
                profile_use[sender["profile_id"]] += 1
                site_use[u] += 1
                zero_links += q_forward == 0
                co_located += u == v
                zero_duration += float(sample["flowDuration"]) == 0
                control += sample["is_control_protocol"] == "1"
                if len(sample_examples) < 3:
                    sample_examples.append(dict(transaction=record, sender=sender,
                                                recipient=recipient, template=sample))
                if rows % 100000 == 0:
                    print(f"transactions={rows} accounts={len(accounts)}", flush=True)
        if not rows:
            raise ValueError("Empty blockchain input")
        write_csv(output / "account_profiles.csv", ACCOUNT_FIELDS,
                  [accounts[key] for key in sorted(accounts)])
        summary = dict(version=VERSION, status="generated_pending_validation", transactions=rows,
            accounts=len(accounts), directed_account_pairs=len(pair_counts),
            original_transaction_sha256=source_sha, transaction_copy_sha256=digest(transactions),
            original_transaction_file=str(args.chain_csv.resolve()), index_offset=args.index_offset,
            seed=args.seed, templates_per_profile=args.templates_per_profile,
            unique_templates_used=len(template_use), repeated_template_assignments=rows-len(template_use),
            max_template_reuse=max(template_use.values()), zero_directed_quality_rows=zero_links,
            co_located_rows=co_located, zero_duration_rows=zero_duration, control_template_rows=control,
            account_profile_counts=dict(Counter(a["profile_id"] for a in accounts.values())),
            account_site_counts=dict(Counter(str(a["site_id"]) for a in accounts.values())),
            transaction_profile_counts=dict(profile_use),
            source=metadata, elapsed_seconds=time.perf_counter()-started,
            temporal_semantics="blockchain file order only; flow time is independent provenance",
            integration_status="dataset_only; old IoT identity/anchor loader must be adapted",
            future_information_used=False, scene_assigned_before_policy=True,
            train_validation_test_split="not selected by generator",
            resource_complexity="streaming in transaction count; memory grows with unique accounts/pairs "
            "and bounded per-profile template pool",
            code_sha256={Path(__file__).name: digest(__file__)})
        write_json(output / "construction_examples.json", sample_examples)
        write_json(output / "template_usage.json", dict(template_use))
        write_json(output / "dataset_summary.json", summary)
        status.update(status="generated_pending_validation", transactions=rows, accounts=len(accounts))
        write_json(output / "build_status.json", status)
        return summary
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        write_json(output / "build_status.json", status)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--chain-csv", type=Path, default=ROOT / "selectedTxs_300K.csv")
    result.add_argument("--iot-source-dir", type=Path, default=ROOT.parent / "原始数据集")
    result.add_argument("--output-dir", type=Path, default=ROOT / "data_iot_v2")
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--templates-per-profile", type=int, default=4096)
    result.add_argument("--reuse-library", type=Path,
                        help="reuse the frozen library in an earlier v2 directory")
    result.add_argument("--index-offset", type=int, default=0,
                        help="global row offset for chunks of one ordered blockchain file")
    return result


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    result = build(parser().parse_args())
    print(json.dumps({k: result[k] for k in ("transactions", "accounts", "unique_templates_used",
                                            "elapsed_seconds")}, ensure_ascii=False, indent=2))
