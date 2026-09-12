"""生成器的性质检查：身份保留、追加稳定、分块一致、损坏拒绝与有向零链路。"""
import contextlib
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_iot_v2_dataset import build, parser, read_topology
from validate_iot_v2_dataset import validate, verify_template_sources


def address(number):
    return f"0x{number:040x}"


def make_chain(path, count, offset=0):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for i in range(offset, offset+count):
            row = [""] * 18
            row[3], row[4], row[6], row[7], row[8] = address(i+1), address((i+5)%13+30), "0", "0", "900719925474099312345"
            writer.writerow(row)


def sources(root):
    source = root / "source"
    source.mkdir()
    (source / "mote_locs.txt").write_text("1 0 0\n2 3 4\n", encoding="utf-8")
    (source / "connectivity.txt").write_text("1 1 0\n1 2 0\n2 1 0.7\n2 2 0\n0 1\n", encoding="utf-8")
    fields = ["time", "protocol", "srcNumPackets", "dstNumPackets",
              "srcPayloadSize", "dstPayloadSize", "flowDuration"]
    with zipfile.ZipFile(source / "flows.zip", "w") as archive:
        for name in ("ToyA_001122334455", "ToyB_66778899aabb"):
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(fields)
            for i in range(8):
                writer.writerow([f"2016-10-01 00:00:0{i}", "dns", 1, 1, 37+i, 85, i])
            writer.writerow(["bad-time", "dns", 1, 1, 37, 85, 0])
            archive.writestr(f"flows/{name}_flows.csv", buffer.getvalue())
    return source


def generate(root, source, name, count=12, offset=0, reuse=None):
    chain = root / f"{name}.csv"
    make_chain(chain, count, offset)
    args = ["--chain-csv", str(chain), "--iot-source-dir", str(source),
            "--output-dir", str(root / name), "--templates-per-profile", "4",
            "--index-offset", str(offset)]
    if reuse:
        args += ["--reuse-library", str(reuse)]
    with contextlib.redirect_stdout(io.StringIO()):
        build(parser().parse_args(args))
    return root / name, chain


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class V2DatasetTests(unittest.TestCase):
    def test_exact_copy_and_complete_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sources(root)
            output, chain = generate(root, source, "small")
            checked = validate(output, chain)
            self.assertEqual(chain.read_bytes(), (output / "selectedTxs_iot_v2.csv").read_bytes())
            self.assertEqual(checked["transactions"], 12)
            self.assertTrue(checked["byte_identical_to_original"])
            self.assertEqual(json.loads((output / "scene_library.json").read_text())["eligible_flows"], 16)
            with contextlib.redirect_stdout(io.StringIO()):
                traced = verify_template_sources(output, source)
            self.assertEqual(traced["templates_checked"], 8)

    def test_append_and_chunk_do_not_remap_existing_accounts_or_templates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sources(root)
            short, _ = generate(root, source, "short", count=5)
            full, _ = generate(root, source, "full", count=12, reuse=short)
            tail, _ = generate(root, source, "tail", count=7, offset=5, reuse=short)
            first = csv_rows(short / "transaction_scene.csv")
            entire = csv_rows(full / "transaction_scene.csv")
            chunk = csv_rows(tail / "transaction_scene.csv")
            self.assertEqual(first, entire[:5])
            for a, b in zip(entire[5:], chunk):
                a.pop("source_row")
                b.pop("source_row")
                self.assertEqual(a, b)
            for dataset in (short, full, tail):
                validate(dataset)

    def test_corrupted_identity_and_template_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sources(root)
            output, _ = generate(root, source, "small")
            path = output / "transaction_scene.csv"
            original = path.read_text(encoding="utf-8")
            path.write_text(original.replace(address(1), address(999), 1), encoding="utf-8")
            with self.assertRaises(AssertionError):
                validate(output)
            path.write_text(original, encoding="utf-8")
            templates = output / "flow_templates.csv"
            templates.write_text(templates.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                validate(output)

    def test_zero_and_asymmetric_links_preserved_and_real_missing_pair_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sources(root)
            _, links, ignored = read_topology(source)
            self.assertEqual(links[1, 2], 0)
            self.assertEqual(links[2, 1], .7)
            self.assertEqual(ignored["incomplete_links_without_coordinates"], 1)
            path = source / "connectivity.txt"
            path.write_text("1 1 0\n1 2 0\n2 2 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_topology(source)

    def test_existing_dataset_and_changed_frozen_library_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sources(root)
            output, chain = generate(root, source, "small")
            with self.assertRaises(FileExistsError):
                build(parser().parse_args(["--chain-csv", str(chain), "--output-dir", str(output)]))
            path = output / "sites.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate(root, source, "reused", reuse=output)


if __name__ == "__main__":
    unittest.main()
