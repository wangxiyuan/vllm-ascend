# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the A3 560T routing logic in select_tests.py.

Accuracy tests (and ready-all full runs) must be scheduled onto dedicated
560T runner labels (``linux-aarch64-a3-800i-N``) instead of the generic
mixed 560T/752T pools (``linux-aarch64-a3-N-``).
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / ".github/workflows/scripts/select_tests.py"

_spec = importlib.util.spec_from_file_location("ci_select_tests", _SCRIPT_PATH)
select_tests = importlib.util.module_from_spec(_spec)
sys.modules["ci_select_tests"] = select_tests
_spec.loader.exec_module(select_tests)

PartitionInfo = select_tests.PartitionInfo


def _default_partition_config() -> dict[str, PartitionInfo]:
    return {
        "a3-2": PartitionInfo(runner_label="linux-aarch64-a3-2-", count=2),
        "a3-4": PartitionInfo(runner_label="linux-aarch64-a3-4-", count=2),
        "a3-8": PartitionInfo(runner_label="linux-aarch64-a3-8-", count=1),
        "a2-1": PartitionInfo(runner_label="linux-aarch64-a2b3-1", count=1),
    }


class TestHasAccuracyTests(unittest.TestCase):
    """Accuracy tests are exactly the files listed in accuracy_tests."""

    _ACCURACY_TESTS = {
        "tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py",
        "tests/e2e/pull_request/two_card/test_shared_expert_dp.py",
    }

    def test_detects_accuracy_test(self):
        groups = {"a3-4": ["tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py"]}
        self.assertTrue(select_tests._has_accuracy_tests(groups, self._ACCURACY_TESTS))

    def test_detects_accuracy_test_in_nodeid(self):
        groups = {"a3-4": ["tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py::test_case"]}
        self.assertTrue(select_tests._has_accuracy_tests(groups, self._ACCURACY_TESTS))

    def test_detects_accuracy_test_in_other_partition(self):
        groups = {"a3-2": ["tests/e2e/pull_request/two_card/test_shared_expert_dp.py"]}
        self.assertTrue(select_tests._has_accuracy_tests(groups, self._ACCURACY_TESTS))

    def test_unlisted_file_is_not_accuracy(self):
        groups = {"a3-2": ["tests/e2e/pull_request/two_card/test_mla_precision.py"]}
        self.assertFalse(select_tests._has_accuracy_tests(groups, self._ACCURACY_TESTS))

    def test_no_accuracy_tests(self):
        groups = {"a3-2": ["tests/e2e/pull_request/two_card/test_activation.py"]}
        self.assertFalse(select_tests._has_accuracy_tests(groups, self._ACCURACY_TESTS))

    def test_empty_groups(self):
        self.assertFalse(select_tests._has_accuracy_tests({}, self._ACCURACY_TESTS))

    def test_load_accuracy_tests_rejects_nodeids(self):
        with self.assertRaises(ValueError):
            select_tests._load_accuracy_tests({"accuracy_tests": ["tests/e2e/test_a.py::test_x"]})

    def test_load_accuracy_tests_from_config(self):
        import yaml

        with open(select_tests._CONFIG_PATH) as f:
            meta = yaml.safe_load(f)
        accuracy_tests = select_tests._load_accuracy_tests(meta)
        self.assertIn("tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py", accuracy_tests)


class TestPrefer560TPartitions(unittest.TestCase):
    def test_swaps_generic_labels_and_keeps_counts(self):
        config = _default_partition_config()
        updated = select_tests._prefer_560t_partitions(config)
        self.assertEqual(updated["a3-2"].runner_label, "linux-aarch64-a3-800i-2")
        self.assertEqual(updated["a3-2"].count, 2)
        self.assertEqual(updated["a3-4"].runner_label, "linux-aarch64-a3-800i-4")
        self.assertEqual(updated["a3-4"].count, 2)
        self.assertEqual(updated["a3-8"].runner_label, "linux-aarch64-a3-800i-8")
        self.assertEqual(updated["a3-8"].count, 1)

    def test_does_not_mutate_input(self):
        config = _default_partition_config()
        select_tests._prefer_560t_partitions(config)
        self.assertEqual(config["a3-2"].runner_label, "linux-aarch64-a3-2-")

    def test_leaves_non_a3_and_dedicated_partitions(self):
        config = _default_partition_config()
        config["a3-800i-4"] = PartitionInfo(runner_label="linux-aarch64-a3-800i-4", count=2)
        updated = select_tests._prefer_560t_partitions(config)
        self.assertEqual(updated["a3-800i-4"].runner_label, "linux-aarch64-a3-800i-4")
        self.assertEqual(updated["a2-1"].runner_label, "linux-aarch64-a2b3-1")


class TestResolveToRunners(unittest.TestCase):
    """End-to-end label resolution against the real runner_label.json."""

    @classmethod
    def setUpClass(cls):
        cls.runners = select_tests._load_runners()
        for label in (
            "linux-aarch64-a3-800i-2",
            "linux-aarch64-a3-800i-4",
            "linux-aarch64-a3-800i-8",
        ):
            assert label in cls.runners, f"{label} must exist in runner_label.json"

    def _runners_for(self, partition_config):
        groups = {"a3-2": ["tests/e2e/pull_request/two_card/test_llama_tp2.py"]}
        return {
            group["partition_name"]: group["runner"]
            for group in select_tests._resolve_to_runners(groups, self.runners, partition_config, {})
        }

    def test_accuracy_run_resolves_to_560t_labels(self):
        config = select_tests._prefer_560t_partitions(_default_partition_config())
        self.assertEqual(self._runners_for(config)["a3-2"], "linux-aarch64-a3-800i-2")

    def test_non_accuracy_run_keeps_mixed_pool(self):
        self.assertEqual(self._runners_for(_default_partition_config())["a3-2"], "linux-aarch64-a3-2-")


class TestPartitionConfigConsistency(unittest.TestCase):
    def test_no_sixteen_card_partition(self):
        with open(select_tests._CONFIG_PATH) as f:
            meta = yaml.safe_load(f)
        self.assertNotIn("a3-16", meta["partition"])
        self.assertNotIn("linux-aarch64-a3-16-", meta["partition"])


if __name__ == "__main__":
    unittest.main()
