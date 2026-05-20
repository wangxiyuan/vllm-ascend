#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Post Smart UT Test Scope summary to GitHub Step Summary.

Reads test groups and matched modules from environment variables,
formats them into a Markdown report, and writes to $GITHUB_STEP_SUMMARY.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml


def main() -> None:
    """Generate and post the Smart UT Test Scope summary."""
    # 1. Load inputs from environment
    test_groups_str = os.environ.get("TEST_GROUPS", "[]")
    matched_modules = os.environ.get("MATCHED_MODULES", "")

    try:
        groups = json.loads(test_groups_str)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse TEST_GROUPS: {e}", file=sys.stderr)
        groups = []

    # 2. Load blacklist
    blacklist_path = Path(__file__).parent / "ut_blacklist.yaml"
    blacklist: list[str] = []
    if blacklist_path.exists():
        try:
            data = yaml.safe_load(blacklist_path.read_text())
            if data:
                blacklist = data
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse blacklist: {e}", file=sys.stderr)

    # 3. Build Markdown report
    lines: list[str] = ["## Smart UT Test Scope", ""]
    lines.append(f"**Matched modules:** {matched_modules}")
    lines.append("")

    if blacklist:
        lines.append("<details>")
        lines.append(f"<summary>Blacklisted ({len(blacklist)} tests)</summary>")
        lines.append("")
        for bl in sorted(blacklist):
            lines.append(f"- `{bl}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    for g in groups:
        npu_type = g.get("npu_type", "unknown")
        num_npus = g.get("num_npus", 0)
        runner = g.get("runner", "unknown")
        tests = g.get("tests", "").split()

        if npu_type == "cpu":
            header = f"### CPU ({len(tests)} tests) \u2192 `{runner}`"
        else:
            header = f"### {npu_type.upper()} x{num_npus} ({len(tests)} tests) \u2192 `{runner}`"

        lines.append(header)
        lines.append("")
        lines.append("| # | Test target |")
        lines.append("|---|------------|")
        for i, t in enumerate(tests, 1):
            lines.append(f"| {i} | `{t}` |")
        lines.append("")

    report = "\n".join(lines)

    # 4. Write to GitHub Step Summary
    github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "a") as f:
            f.write(report)
        print("Summary posted successfully.", file=sys.stderr)
    else:
        # Fallback for local testing
        print(report)


if __name__ == "__main__":
    main()
