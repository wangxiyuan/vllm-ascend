#!/usr/bin/env python3
"""Merge curated upstream (vLLM checkout) test groups into the main selection.

Reads the ``GITHUB_OUTPUT`` files produced by two ``select_tests.py`` runs
(the main selection and ``--curated upstream_pr``), appends the upstream
targets to an existing ``a2-1`` group so they piggyback on that machine, and
rewrites the merged ``has_tests`` / ``test_groups`` / ``csrc_cache_target_ids``
outputs. Falls back to a single upstream-only group when the main selection
produced no ``a2-1`` group.

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


def main() -> None:
    main_path, upstream_path = sys.argv[1], sys.argv[2]
    rec = read_outputs(main_path)
    upstream = read_outputs(upstream_path)

    groups = json.loads(rec.get("test_groups", "[]"))
    upstream_groups = json.loads(upstream.get("test_groups", "[]"))
    targets = [t for g in upstream_groups for t in g.get("tests", "").split()]

    if targets:
        host = next((g for g in groups if g.get("partition_name") == UPSTREAM_PARTITION), None)
        if host is not None:
            host["tests"] = (host.get("tests", "") + " " + " ".join(targets)).strip()
        else:
            fallback = dict(upstream_groups[0])
            fallback["tests"] = " ".join(targets)
            groups.append(fallback)

    cache_targets = sorted(
        set(json.loads(rec.get("csrc_cache_target_ids", "[]")))
        | set(json.loads(upstream.get("csrc_cache_target_ids", "[]")))
    )
    merged = {
        "has_tests": str(len(groups) > 0).lower(),
        "test_groups": json.dumps(groups, separators=(",", ":")),
        "csrc_cache_target_ids": json.dumps(cache_targets, separators=(",", ":")),
    }
    with open(main_path, "a", encoding="utf-8") as out:
        for key, value in merged.items():
            print(f"{key}={value}", file=out)
    print(f"Merged test groups: {len(groups)} (upstream targets appended: {len(targets)})")


if __name__ == "__main__":
    main()
