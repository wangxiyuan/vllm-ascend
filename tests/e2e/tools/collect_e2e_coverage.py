import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "tests" / "e2e" / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "e2e_coverage_map.json"
DEFAULT_TESTS = ["tests/e2e/pull_request"]
SOURCE_PACKAGE = "vllm_ascend"


def repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check_coverage_available() -> None:
    if shutil.which(sys.executable) is None:
        raise RuntimeError(f"Python executable not found: {sys.executable}")
    result = subprocess.run(
        [sys.executable, "-c", "import coverage"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "coverage.py is not installed in this environment. Install coverage or pytest-cov before running Phase 2."
        )


def write_coveragerc(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[run]",
                "branch = True",
                f"source = {SOURCE_PACKAGE}",
                "dynamic_context = test_function",
                "context = ${COVERAGE_CONTEXT-}",
                "parallel = True",
                "",
                "[report]",
                "include = vllm_ascend/*",
                "",
                "[json]",
                "show_contexts = True",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_subprocess_support(work_dir: Path) -> None:
    (work_dir / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import os",
                "",
                "if os.environ.get('COVERAGE_PROCESS_START'):",
                "    try:",
                "        import coverage",
                "        coverage.process_startup()",
                "    except Exception:",
                "        pass",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (work_dir / "e2e_coverage_context.py").write_text(
        "\n".join(
            [
                "import os",
                "",
                "def pytest_runtest_setup(item):",
                "    os.environ['COVERAGE_CONTEXT'] = item.nodeid",
                "",
                "def pytest_runtest_teardown(item, nextitem):",
                "    os.environ.pop('COVERAGE_CONTEXT', None)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def run_coverage(
    tests: list[str], pytest_args: list[str], work_dir: Path, allow_test_failures: bool
) -> tuple[Path, int]:
    coveragerc = work_dir / ".coveragerc"
    coverage_file = work_dir / ".coverage"
    coverage_json = work_dir / "coverage.json"
    write_coveragerc(coveragerc)
    write_subprocess_support(work_dir)
    env = os.environ.copy()
    env["COVERAGE_RCFILE"] = str(coveragerc)
    env["COVERAGE_FILE"] = str(coverage_file)
    env["COVERAGE_PROCESS_START"] = str(coveragerc)
    env["PYTHONPATH"] = f"{work_dir}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(work_dir)
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "-m",
        "pytest",
        "-p",
        "e2e_coverage_context",
        *tests,
        *pytest_args,
    ]
    pytest_result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, check=False)
    if pytest_result.returncode != 0 and not allow_test_failures:
        raise RuntimeError(f"Command failed with exit code {pytest_result.returncode}: {' '.join(command)}")
    run_command([sys.executable, "-m", "coverage", "combine", "--keep", str(work_dir)], env)
    run_command([sys.executable, "-m", "coverage", "json", "--show-contexts", "-o", str(coverage_json)], env)
    return coverage_json, pytest_result.returncode


def normalize_context(context: str) -> str:
    if not context:
        return "<no_context>"
    return context.removesuffix("|run").removesuffix("|setup").removesuffix("|teardown")


def merge_line_map(target: dict[str, list[int]], source: dict[str, list[int]]) -> None:
    for file, lines in source.items():
        target[file] = sorted(set(target.get(file, [])) | set(lines))


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_maps(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if not old:
        return new
    merged = new.copy()
    merged["tests"] = old.get("tests", {}).copy()
    merged["source_files"] = old.get("source_files", {}).copy()

    for test, data in new.get("tests", {}).items():
        if test not in merged["tests"]:
            merged["tests"][test] = data
            continue
        old_data = merged["tests"][test]
        old_data["source_files"] = sorted(set(old_data.get("source_files", [])) | set(data.get("source_files", [])))
        merge_line_map(old_data.setdefault("source_lines", {}), data.get("source_lines", {}))

    for source_file, data in new.get("source_files", {}).items():
        if source_file not in merged["source_files"]:
            merged["source_files"][source_file] = data
            continue
        old_data = merged["source_files"][source_file]
        old_data["tests"] = sorted(set(old_data.get("tests", [])) | set(data.get("tests", [])))
        old_data["covered_lines"] = sorted(set(old_data.get("covered_lines", [])) | set(data.get("covered_lines", [])))

    merged["metadata"] = new.get("metadata", {})
    merged["metadata"]["append_merged"] = True
    return merged


def parse_coverage_json(path: Path, tests: list[str], pytest_args: list[str]) -> dict[str, Any]:
    coverage_data = json.loads(path.read_text(encoding="utf-8"))
    by_test: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    unattributed: dict[str, list[int]] = defaultdict(list)

    for file_name, file_data in coverage_data.get("files", {}).items():
        rel_file = repo_relative(file_name)
        if not rel_file.startswith(f"{SOURCE_PACKAGE}/"):
            continue
        contexts = file_data.get("contexts", {})
        for line, line_contexts in contexts.items():
            try:
                line_no = int(line)
            except ValueError:
                continue
            if not line_contexts:
                unattributed[rel_file].append(line_no)
                continue
            for raw_context in line_contexts:
                context = normalize_context(raw_context)
                if context == "<no_context>":
                    unattributed[rel_file].append(line_no)
                    continue
                test_data = by_test.setdefault(context, {"source_files": [], "source_lines": {}})
                test_data["source_lines"].setdefault(rel_file, []).append(line_no)
                source_data = by_source.setdefault(rel_file, {"tests": [], "covered_lines": []})
                source_data["tests"].append(context)
                source_data["covered_lines"].append(line_no)

    for test_data in by_test.values():
        test_data["source_files"] = sorted(test_data["source_lines"])
        for file, lines in test_data["source_lines"].items():
            test_data["source_lines"][file] = sorted(set(lines))
    for source_data in by_source.values():
        source_data["tests"] = sorted(set(source_data["tests"]))
        source_data["covered_lines"] = sorted(set(source_data["covered_lines"]))

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "source": SOURCE_PACKAGE,
            "tests_arg": tests,
            "pytest_args": pytest_args,
            "coverage_format": coverage_data.get("meta", {}).get("format"),
            "coverage_version": coverage_data.get("meta", {}).get("version"),
        },
        "tests": dict(sorted(by_test.items())),
        "source_files": dict(sorted(by_source.items())),
        "unattributed_lines": {file: sorted(set(lines)) for file, lines in sorted(unattributed.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect E2E coverage map with coverage.py dynamic contexts.")
    parser.add_argument("--tests", nargs="+", default=DEFAULT_TESTS, help="pytest test paths or nodeids to run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output coverage map JSON path")
    parser.add_argument("--append", action="store_true", help="merge with existing output JSON")
    parser.add_argument("--keep-raw", action="store_true", help="keep raw coverage working directory")
    parser.add_argument(
        "--allow-test-failures",
        action="store_true",
        help="write coverage output even if pytest fails and return success",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="extra pytest args after --")
    return parser.parse_args()


def normalize_pytest_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    args = parse_args()
    pytest_args = normalize_pytest_args(args.pytest_args)
    try:
        check_coverage_available()
        with tempfile.TemporaryDirectory(prefix="e2e-coverage-", dir="/tmp") as temp_dir:
            work_dir = Path(temp_dir)
            coverage_json, pytest_exit_code = run_coverage(args.tests, pytest_args, work_dir, args.allow_test_failures)
            result = parse_coverage_json(coverage_json, args.tests, pytest_args)
            result["metadata"]["pytest_exit_code"] = pytest_exit_code
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.append:
                result = merge_maps(load_existing(output), result)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if args.keep_raw:
                raw_dir = output.parent / "raw_e2e_coverage"
                if raw_dir.exists():
                    shutil.rmtree(raw_dir)
                shutil.copytree(work_dir, raw_dir)
                result["metadata"]["raw_dir"] = repo_relative(raw_dir)
                output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote coverage map to {repo_relative(args.output)}")
        return 0
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
