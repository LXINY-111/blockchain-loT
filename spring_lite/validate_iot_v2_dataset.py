"""独立核对 v2 数据的交易保留、外键、映射、模板和有向链路，不启动系统实验。"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def keyed(path, key):
    result = {}
    for row in rows(path):
        if row[key] in result:
            raise AssertionError(f"Duplicate {key} in {path}")
        result[row[key]] = row
    return result


def hash_choice(version, seed, purpose, *parts):
    encoded = json.dumps([version, seed, purpose, *parts], ensure_ascii=True,
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


def validate(directory, source=None):
    if not __debug__:
        raise RuntimeError("Validation must run without Python -O; assertions are required")
    directory = Path(directory)
    summary = json.loads((directory / "dataset_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "scene_library.json").read_text(encoding="utf-8"))
    source = Path(source or summary["original_transaction_file"])
    copied = directory / "selectedTxs_iot_v2.csv"
    assert digest(source) == digest(copied) == summary["original_transaction_sha256"], "Transaction bytes changed"
    for name, expected in metadata["library_file_sha256"].items():
        assert digest(directory / name) == expected, f"Changed library {name}"
    accounts = keyed(directory / "account_profiles.csv", "account_address")
    profiles = keyed(directory / "device_profiles.csv", "profile_id")
    templates = keyed(directory / "flow_templates.csv", "template_id")
    sites = {int(r["site_id"]): r for r in rows(directory / "sites.csv")}
    links = {(int(r["from_site_id"]), int(r["to_site_id"])): r
             for r in rows(directory / "site_links.csv")}
    assert len(links) == len(sites) ** 2, "Directed topology incomplete"
    pools = {p: [] for p in profiles}
    for sample in templates.values():
        assert sample["profile_id"] in profiles
        assert sample["source_file"] == profiles[sample["profile_id"]]["source_file"]
        pools[sample["profile_id"]].append(sample["template_id"])
    seed, version = summary["seed"], summary["version"]
    for address, row in accounts.items():
        assert row["actor_type"] == "device"
        pid = list(profiles)[hash_choice(version, seed, "profile", address) % len(profiles)]
        site = sorted(sites)[hash_choice(version, seed, "site", address) % len(sites)]
        assert row["profile_id"] == pid and int(row["site_id"]) == site, "Unstable account assignment"
        assert float(row["x_m"]) == float(sites[site]["x_m"]) and float(row["y_m"]) == float(sites[site]["y_m"])
    seen, first_seen, pairs, exact_rows, uses = set(), {}, set(), set(), Counter()
    count = duplicate_rows = zero_links = local_rows = zero_duration = zero_payload = 0
    duration_max, distance_max = 0.0, 0.0
    with copied.open(encoding="utf-8-sig", newline="") as chain, (
        directory / "transaction_scene.csv").open(encoding="utf-8", newline="") as scene:
        companion = csv.DictReader(scene)
        for local_index, tx in enumerate(csv.reader(chain)):
            record = next(companion, None)
            assert record is not None, "Scene ended early"
            assert len(tx) == 18 and tx[6:8] == ["0", "0"]
            assert tx[8].isdigit() and tx[3] != tx[4]
            global_index = summary["index_offset"] + local_index
            assert int(record["tx_index"]) == global_index
            assert int(record["source_row"]) == local_index + 1
            a, b = tx[3], tx[4]
            assert (record["from_address"], record["to_address"]) == (a, b), "Address reversal/identity rewrite"
            assert a in accounts and b in accounts, "Orphan account"
            for prefix, address in (("from", a), ("to", b)):
                mapped = accounts[address]
                assert record[prefix + "_profile_id"] == mapped["profile_id"]
                assert record[prefix + "_site_id"] == mapped["site_id"]
                assert record[prefix + "_actor_type"] == mapped["actor_type"]
                seen.add(address)
                first_seen.setdefault(address, global_index)
            sample = templates[record["template_id"]]
            pool = pools[accounts[a]["profile_id"]]
            chosen = pool[hash_choice(version, seed, "transaction_template", global_index, tx) % len(pool)]
            assert record["template_id"] == chosen, "Template does not follow fixed selection rule"
            assert sample["profile_id"] == accounts[a]["profile_id"]
            for field in ("protocol", "srcNumPackets", "dstNumPackets", "srcPayloadSize",
                          "dstPayloadSize", "flowDuration", "is_control_protocol"):
                assert record[field] == sample[field], f"Mixed/corrupted template field: {field}"
            duration = float(record["flowDuration"])
            payload = int(record["srcPayloadSize"]) + int(record["dstPayloadSize"])
            assert math.isfinite(duration) and duration >= 0 and payload >= 0
            u, v = int(record["from_site_id"]), int(record["to_site_id"])
            q, reverse = float(record["link_quality_forward"]), float(record["link_quality_reverse"])
            assert q == float(links[u, v]["effective_quality"])
            assert reverse == float(links[v, u]["effective_quality"])
            assert record["link_quality_rule"] == links[u, v]["rule"]
            expected_distance = math.hypot(float(sites[u]["x_m"]) - float(sites[v]["x_m"]),
                                           float(sites[u]["y_m"]) - float(sites[v]["y_m"]))
            assert math.isclose(float(record["distance_m"]), expected_distance, abs_tol=1e-9)
            if u == v:
                assert q == 1 and reverse == 1 and expected_distance == 0
            else:
                assert q == float(links[u, v]["measured_quality"]), "Measured zero/probability changed"
            signature = hashlib.sha256(json.dumps(tx, separators=(",", ":")).encode()).digest()
            duplicate_rows += signature in exact_rows
            exact_rows.add(signature)
            pairs.add((a, b))
            uses[record["template_id"]] += 1
            count += 1
            zero_links += q == 0
            local_rows += u == v
            zero_duration += duration == 0
            zero_payload += payload == 0
            duration_max, distance_max = max(duration_max, duration), max(distance_max, expected_distance)
        assert next(companion, None) is None, "Extra scene records"
    assert seen == set(accounts), "Extra/missing account rows"
    assert all(int(accounts[a]["first_tx_index"]) == n for a, n in first_seen.items())
    assert (count, len(accounts), len(pairs)) == (summary["transactions"], summary["accounts"],
                                                summary["directed_account_pairs"])
    assert dict(uses) == json.loads((directory / "template_usage.json").read_text(encoding="utf-8"))
    result = dict(status="passed", transactions=count, accounts=len(accounts),
        directed_account_pairs=len(pairs), byte_identical_to_original=True,
        exact_duplicate_rows_preserved=duplicate_rows, unique_templates_used=len(uses),
        repeated_template_assignments=count-len(uses), maximum_template_reuse=max(uses.values()),
        zero_directed_quality_rows=zero_links, co_located_rows=local_rows,
        zero_duration_rows=zero_duration, zero_payload_rows=zero_payload,
        maximum_duration_raw=duration_max, maximum_distance_m=distance_max,
        all_account_and_template_joins_complete=True, deterministic_mapping_checked_every_row=True,
        transaction_order_and_addresses_preserved=True, future_blockchain_statistics_used=False,
        role_counts=dict(Counter(a["actor_type"] for a in accounts.values())),
        checks_scope="dataset structure/content only; no Go/Python integration or PPO performance claim",
        validator_sha256=digest(__file__))
    status = json.loads((directory / "build_status.json").read_text(encoding="utf-8"))
    (directory / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                                               encoding="utf-8")
    summary["status"] = "validated"
    summary["validation_file"] = "validation.json"
    (directory / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
                                                    encoding="utf-8")
    status["status"] = "validated"
    (directory / "build_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n",
                                                 encoding="utf-8")
    return result


def verify_template_sources(directory, source_directory):
    """回读原始 ZIP，逐条核对样本确实来自记录的文件/行，而非仅自我比较。"""
    if not __debug__:
        raise RuntimeError("Source verification must run without -O")
    directory, source_directory = Path(directory), Path(source_directory)
    library = json.loads((directory / "scene_library.json").read_text(encoding="utf-8"))
    for name, expected in library["source_sha256"].items():
        assert digest(source_directory / name) == expected, f"Original source changed: {name}"
    targets = {}
    for record in rows(directory / "flow_templates.csv"):
        targets.setdefault(record["source_file"], {})[int(record["source_row"])] = record
    checked = 0
    with zipfile.ZipFile(source_directory / "flows.zip") as archive:
        for filename, wanted in targets.items():
            found = set()
            with archive.open(filename) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                for number, original in enumerate(reader, 1):
                    if number not in wanted:
                        continue
                    record = wanted[number]
                    for field in ("srcIp", "dstIp", "srcPort", "dstPort", "srcNumPackets",
                                  "dstNumPackets", "srcPayloadSize", "dstPayloadSize", "flowDuration"):
                        assert record[field] == original.get(field, "").strip(), f"Source mismatch {filename}:{number}:{field}"
                    assert record["protocol"] == (original["protocol"].strip().lower() or "none")
                    assert record["time"] == datetime.fromisoformat(original["time"].strip()).isoformat(sep=" ")
                    found.add(number)
                    checked += 1
            assert found == set(wanted), f"Missing source rows in {filename}"
            print(f"source checked: {filename} samples={len(found)}", flush=True)
    result = dict(status="passed", templates_checked=checked, source_files_checked=len(targets),
                  original_source_hashes_match=True, every_template_matches_original_file_and_row=True)
    (directory / "source_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data_iot_v2")
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--verify-template-sources", type=Path, metavar="IOT_SOURCE_DIR",
                        help="also reread original ZIP and verify every retained traffic template")
    args = parser.parse_args()
    try:
        checked = validate(args.dataset_dir, args.source_csv)
        if args.verify_template_sources:
            checked["source_validation"] = verify_template_sources(args.dataset_dir, args.verify_template_sources)
    except Exception as exc:
        # 校验失败时使旧的通过标记失效，避免把上一次检查结果误认为当前结果。
        failed = dict(status="failed", error=repr(exc))
        if args.dataset_dir.is_dir():
            (args.dataset_dir / "validation.json").write_text(
                json.dumps(failed, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        raise
    print(json.dumps(checked, ensure_ascii=False, indent=2))
