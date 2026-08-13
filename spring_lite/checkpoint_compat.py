from math import isclose
from typing import Any, Dict, List, Mapping


def checkpoint_config_mismatches(
    extra: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> List[str]:
    """Return reproducibility-critical checkpoint metadata mismatches."""
    mismatches: List[str] = []
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
