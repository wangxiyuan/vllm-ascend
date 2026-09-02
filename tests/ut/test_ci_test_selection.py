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
"""Unit tests for the CI test selection scripts.

Covers ``merge_upstream_tests.py`` (merging curated upstream ``vllm://``
targets into the main selection outputs) and the ``vllm://`` handling of
``select_tests.py`` / ``assemble_coverage.py``.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "scripts"


def _load_module(name: str):
    """Import a CI script as a module despite its non-package directory."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses in these scripts call sys.modules.get(__module__).__dict__,
    # which returns None unless the module is registered.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


merge_upstream_tests = _load_module("merge_upstream_tests")


def _write_outputs(path: Path, **values: str) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")
    return path


def _read_outputs(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key:
            values[key] = value
    return values


def _host_group() -> dict:
    return {
        "num_npus": 1,
        "npu_type": "a2",
        "runner": "linux-aarch64-a2b3-1",
        "tests": "tests/e2e/pull_request/one_card/test_basic.py",
        "partition_name": "a2-1",
        "partition": "2-5",
    }


def _upstream_group(*tests: str) -> dict:
    return {
        "num_npus": 1,
        "npu_type": "a2",
        "runner": "linux-aarch64-a2b3-1",
        "tests": " ".join(tests),
        "partition_name": "a2-1",
        "partition": "1-5",
    }


class TestMergeUpstreamTests(unittest.TestCase):
    """Tests for merge_upstream_tests.merge_groups and its CLI."""

    def test_append_to_existing_host_group(self):
        host = _host_group()
        groups, appended = merge_upstream_tests.merge_groups(
            [host],
            [_upstream_group("tests/lora/test_lora_manager.py", "tests/v1/worker/test_mamba_utils.py")],
        )
        self.assertEqual(appended, 2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["tests"],
            "tests/e2e/pull_request/one_card/test_basic.py "
            "tests/lora/test_lora_manager.py tests/v1/worker/test_mamba_utils.py",
        )
        # The host group keeps its own partition label.
        self.assertEqual(groups[0]["partition"], "2-5")

    def test_fallback_creates_single_partition_group(self):
        upstream = _upstream_group("tests/lora/test_lora_manager.py")
        groups, appended = merge_upstream_tests.merge_groups([], [upstream])
        self.assertEqual(appended, 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["tests"], "tests/lora/test_lora_manager.py")
        # A lone fallback group runs by itself; a split label would be misleading.
        self.assertEqual(groups[0]["partition"], "1-1")

    def test_no_upstream_targets_keeps_groups_untouched(self):
        host = _host_group()
        groups, appended = merge_upstream_tests.merge_groups([host], [])
        self.assertEqual(appended, 0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["tests"], "tests/e2e/pull_request/one_card/test_basic.py")

    def test_cli_rewrites_main_outputs_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            main_path = _write_outputs(
                Path(tmp) / "main.outputs",
                has_tests="true",
                test_groups=json.dumps([_host_group()], separators=(",", ":")),
                csrc_cache_target_ids='["a2-arm64-ubuntu"]',
                matched_modules="a2,cpu",
            )
            upstream_path = _write_outputs(
                Path(tmp) / "upstream.outputs",
                has_tests="true",
                test_groups=json.dumps([_upstream_group("tests/lora/test_lora_manager.py")], separators=(",", ":")),
                csrc_cache_target_ids='["a2-arm64-ubuntu"]',
                matched_modules="upstream_pr",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPTS_DIR / "merge_upstream_tests.py"),
                    str(main_path),
                    str(upstream_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("upstream targets appended: 1", result.stdout)

            merged = _read_outputs(main_path)
            groups = json.loads(merged["test_groups"])
            # Exactly one test_groups line: the file is rewritten, not appended to,
            # so GitHub Actions cannot pick up a stale duplicate key.
            self.assertEqual(
                len(main_path.read_text(encoding="utf-8").splitlines()),
                len(merged),
            )
            self.assertEqual(merged["has_tests"], "true")
            self.assertEqual(len(groups), 1)
            self.assertIn("tests/lora/test_lora_manager.py", groups[0]["tests"])
            self.assertEqual(merged["csrc_cache_target_ids"], '["a2-arm64-ubuntu"]')
            # matched_modules from both sides are preserved.
            self.assertEqual(merged["matched_modules"], "a2,cpu,upstream_pr")

    def test_cli_rejects_wrong_argument_count(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "merge_upstream_tests.py")],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stderr)


class TestSelectTestsVllmTargets(unittest.TestCase):
    """Tests for select_tests.py handling of ``vllm://`` curated targets."""

    @classmethod
    def setUpClass(cls):
        cls.select_tests = _load_module("select_tests")

    def test_curated_vllm_suite_gets_prefix_and_partition(self):
        meta = {
            "curated_tests": {
                "upstream_pr": {
                    "repo": "vllm",
                    "tests": ["tests/lora/test_lora_manager.py", "tests/v1/worker/test_mamba_utils.py"],
                }
            }
        }
        suites = self.select_tests._load_curated_tests(meta)
        self.assertEqual(
            suites["upstream_pr"],
            [
                "vllm://tests/lora/test_lora_manager.py",
                "vllm://tests/v1/worker/test_mamba_utils.py",
            ],
        )

    def test_curated_plain_list_unchanged(self):
        meta = {"curated_tests": {"a5": ["tests/e2e/pull_request/four_card/test_data_parallel_tp2.py"]}}
        suites = self.select_tests._load_curated_tests(meta)
        self.assertEqual(suites["a5"], ["tests/e2e/pull_request/four_card/test_data_parallel_tp2.py"])

    def test_curated_unsupported_repo_rejected(self):
        meta = {"curated_tests": {"bad": {"repo": "torch", "tests": ["x.py"]}}}
        with self.assertRaises(ValueError):
            self.select_tests._load_curated_tests(meta)

    def test_lookup_estimated_time_strips_vllm_prefix(self):
        estimates = {"tests/lora/test_lora_manager.py": 20}
        self.assertEqual(
            self.select_tests._lookup_estimated_time("vllm://tests/lora/test_lora_manager.py", estimates),
            20,
        )
        self.assertEqual(
            self.select_tests._lookup_estimated_time(
                "vllm://tests/lora/test_lora_manager.py::Foo::test_bar", estimates
            ),
            20,
        )
        self.assertEqual(self.select_tests._lookup_estimated_time("vllm://tests/unknown.py", estimates), 600.0)


class TestAssembleCoverageKey(unittest.TestCase):
    """The coverage key must match what run_selected_tests.sh derives on disk."""

    @classmethod
    def setUpClass(cls):
        cls.assemble_coverage = _load_module("assemble_coverage")

    def test_coverage_key_strips_vllm_prefix(self):
        self.assertEqual(
            self.assemble_coverage.coverage_key("vllm://tests/v1/worker/test_mamba_utils.py"),
            "tests__v1__worker__test_mamba_utils",
        )

    def test_coverage_key_plain_and_nodeid_unchanged(self):
        self.assertEqual(
            self.assemble_coverage.coverage_key("tests/e2e/pull_request/one_card/test_basic.py"),
            "tests__e2e__pull_request__one_card__test_basic",
        )
        self.assertEqual(
            self.assemble_coverage.coverage_key("tests/ut/test_foo.py::TestClass::test_method"),
            "tests__ut__test_foo.py--TestClass--test_method",
        )


if __name__ == "__main__":
    unittest.main()
