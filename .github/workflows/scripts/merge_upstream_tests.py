#!/usr/bin/env python3
"""Merge curated upstream (vLLM checkout) test groups into the main selection.

Reads the ``GITHUB_OUTPUT`` files produced by two ``select_tests.py`` runs
(the main selection and ``--curated upstream_pr``), appends the upstream
targets to an existing ``a2-1`` group so they piggyback on that machine, and
rewrites the merged ``has_tests`` / ``test_groups`` / ``csrc_cache_target_ids``
/ ``matched_modules`` outputs back into the main-outputs file (in place, no
duplicate keys are left behind). Falls back to a single upstream-only group
when the main selection produced no ``a2-1`` group.

Usage: merge_upstream_tests.py <main-outputs-file> <upstream-outputs-file>
"""

import json
import sys

UPSTREAM_PARTITION = "a2-1"


def read_outputs(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key:
                values[key] = value
    return values


def merge_groups(main_groups: list, upstream_groups: list) -> tuple[list, int]:
    """Append upstream targets to the a2-1 host group or add a fallback group.

    Returns ``(merged_groups, appended_target_count)``. The fallback group is
    a single-partition group ("1-1"): it runs every upstream target alone, so
    the split label copied from the upstream selection would be misleading.
    """
    targets = [t for g in upstream_groups for t in g.get("tests", "").split()]
    if targets:
        host = next((g for g in main_groups if g.get("partition_name") == UPSTREAM_PARTITION), None)
        if host is not None:
            host["tests"] = (host.get("tests", "") + " " + " ".join(targets)).strip()
        else:
            fallback = dict(upstream_groups[0])
            fallback["tests"] = " ".join(targets)
            fallback["partition"] = "1-1"
            main_groups.append(fallback)
    return main_groups, len(targets)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <main-outputs-file> <upstream-outputs-file>", file=sys.stderr)
        sys.exit(1)
    main_path, upstream_path = sys.argv[1], sys.argv[2]
    rec = read_outputs(main_path)
    upstream = read_outputs(upstream_path)

    groups, appended = merge_groups(
        json.loads(rec.get("test_groups", "[]")),
        json.loads(upstream.get("test_groups", "[]")),
    )
    cache_targets = sorted(
        set(json.loads(rec.get("csrc_cache_target_ids", "[]")))
        | set(json.loads(upstream.get("csrc_cache_target_ids", "[]")))
    )
    matched_modules = sorted(
        {m for m in rec.get("matched_modules", "").split(",") if m}
        | {m for m in upstream.get("matched_modules", "").split(",") if m}
    )

    rec["has_tests"] = str(len(groups) > 0).lower()
    rec["test_groups"] = json.dumps(groups, separators=(",", ":"))
    rec["csrc_cache_target_ids"] = json.dumps(cache_targets, separators=(",", ":"))
    rec["matched_modules"] = ",".join(matched_modules)

    with open(main_path, "w", encoding="utf-8") as out:
        for key, value in rec.items():
            print(f"{key}={value}", file=out)
    print(f"Merged test groups: {len(groups)} (upstream targets appended: {appended})")


if __name__ == "__main__":
    main()
