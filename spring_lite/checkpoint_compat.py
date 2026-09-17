from math import isclose
from typing import Any, Dict, List, Mapping
from pathlib import Path


def resolve_checkpoint_path(manifest_path, recorded_path):
    """优先解析新相对路径；旧绝对路径失效时只查清单附近的明确位置。

    原始归档保持只读，不扫描磁盘、不按模糊文件名猜测模型。
    """
    manifest = Path(manifest_path).resolve()
    source = Path(recorded_path)
    if not source.is_absolute():
        source = manifest.parent / source
    if source.is_file():
        return source.resolve()
    candidates = [manifest.parent / Path(recorded_path).name,
                  manifest.parent.parent / Path(recorded_path).name]
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(found) != 1:
        raise FileNotFoundError(f'Missing or ambiguous relocated checkpoint: {recorded_path}')
    return found[0]


def checkpoint_config_mismatches(
    extra: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> List[str]:
    """Return reproducibility-critical checkpoint metadata mismatches."""
    mismatches: List[str] = []
    if extra.get('mechanism_version', 'legacy') != expected.get('mechanism_version', 'legacy'):
        mismatches.append('mechanism_version: legacy and v3 checkpoints cannot be mixed')
    for key, expected_value in expected.items():
        if key not in extra:
            mismatches.append(f"{key}: missing (expected {expected_value!r})")
            continue

        actual_value = extra[key]
        if isinstance(expected_value, float):
            try:
                matches = isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value

        if not matches:
            mismatches.append(
                f"{key}: checkpoint={actual_value!r}, expected={expected_value!r}"
            )
    return mismatches


def format_checkpoint_mismatch(mismatches: List[str]) -> str:
    return "checkpoint config mismatch: " + "; ".join(mismatches)
