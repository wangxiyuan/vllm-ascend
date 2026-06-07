import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import select_tests
import yaml

CONFIG_PATH = Path(".github/workflows/scripts/test_config.yaml")
E2E_PR_ROOT = Path("tests/e2e/pull_request")
UT_ROOT = Path("tests/ut")
SOURCE_ROOT = Path("vllm_ascend")
GENERATED_SOURCE_FILES = {
    "vllm_ascend/_build_info.py",
    "vllm_ascend/_version.py",
}


class CheckResult:
    def __init__(self) -> None:
        self.errors: dict[str, list[str]] = defaultdict(list)
        self.warnings: dict[str, list[str]] = defaultdict(list)

    def error(self, category: str, message: str) -> None:
        self.errors[category].append(message)

    def warning(self, category: str, message: str) -> None:
        self.warnings[category].append(message)

    def has_errors(self) -> bool:
        return any(self.errors.values())


def load_config(result: CheckResult) -> list[dict[str, Any]]:
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text())
    except Exception as error:
        result.error("YAML parse errors", str(error))
        return []
    if not isinstance(raw, list):
        result.error("YAML schema errors", "test_config.yaml must contain a list of modules")
        return []
    return raw


def pytest_node_file_path(path: str) -> str:
    return select_tests._pytest_node_file_path(path)


def normalized(path: str) -> str:
    return path.rstrip("/")


def path_exists(path: str) -> bool:
    return Path(pytest_node_file_path(normalized(path))).exists()


def expand_test_path(path: str) -> list[str]:
    base = Path(pytest_node_file_path(normalized(path)))
    if not base.exists():
        return []
    if base.is_file():
        return [normalized(path)]
    return sorted(str(file) for file in base.rglob("test_*.py"))


def check_duplicate_modules(config: list[dict[str, Any]], result: CheckResult) -> None:
    names = [module.get("name") for module in config]
    for name, count in sorted(Counter(names).items()):
        if name is None:
            result.error("Module schema errors", "module missing required field: name")
        elif count > 1:
            result.error("Duplicate module names", str(name))


def resolve_config(config: list[dict[str, Any]], result: CheckResult) -> list[dict[str, Any]]:
    try:
        return select_tests._resolve_config_inheritance(config)
    except ValueError as error:
        result.error("Config inheritance errors", str(error))
        return []


def check_paths(config: list[dict[str, Any]], result: CheckResult) -> None:
    for module in config:
        module_name = module.get("name", "<unknown>")
        for field in ("tests", "skip_tests", "source_file_dependencies", "exclude_source_file_dependencies"):
            for path in module.get(field, []):
                if (
                    field in {"source_file_dependencies", "exclude_source_file_dependencies"}
                    and normalized(path) in GENERATED_SOURCE_FILES
                ):
                    continue
                if not path_exists(path):
                    result.error(f"Missing {field} paths", f"{module_name}: {path}")
        for test_path in module.get("tests", []):
            expanded = expand_test_path(test_path)
            if not expanded:
                result.error("Empty test path expansions", f"{module_name}: {test_path}")


def check_e2e_routing(config: list[dict[str, Any]], result: CheckResult) -> None:
    for module in config:
        module_name = module.get("name", "<unknown>")
        for test_path in module.get("tests", []):
            target = normalized(test_path)
            if not select_tests._is_e2e_path(target):
                continue
            path = Path(pytest_node_file_path(target))
            if not path.exists():
                continue
            if path.is_file():
                if select_tests._route_e2e_file(target) is None:
                    result.error("Invalid E2E runner routing", f"{module_name}: {target}")
                continue
            expanded = expand_test_path(target)
            if expanded and all(select_tests._route_e2e_file(file) is None for file in expanded):
                result.error("Invalid E2E runner routing", f"{module_name}: {target}")


def configured_tests(config: list[dict[str, Any]], prefix: str) -> set[str]:
    resolved = set()
    skip_tests = {normalized(path) for module in config for path in module.get("skip_tests", [])}
    for module in config:
        for test_path in module.get("tests", []):
            target = normalized(test_path)
            if not target.startswith(prefix):
                continue
            for expanded in expand_test_path(target):
                if not select_tests._is_skipped_test_target(expanded, skip_tests):
                    resolved.add(pytest_node_file_path(expanded))
    return resolved


def check_test_coverage(config: list[dict[str, Any]], result: CheckResult) -> None:
    configured_e2e = configured_tests(config, "tests/e2e/")
    actual_e2e = {str(file) for file in E2E_PR_ROOT.rglob("test_*.py")}
    for path in sorted(actual_e2e - configured_e2e):
        result.error("Unconfigured E2E pull_request tests", path)

    configured_ut = configured_tests(config, "tests/ut/")
    actual_ut = {str(file) for file in UT_ROOT.rglob("test_*.py")}
    for path in sorted(actual_ut - configured_ut):
        result.error("Unconfigured UT tests", path)


def check_source_coverage(config: list[dict[str, Any]], result: CheckResult) -> None:
    source_deps = {normalized(dep) for module in config for dep in module.get("source_file_dependencies", [])}
    covered_source = set()
    for file in SOURCE_ROOT.rglob("*.py"):
        source_file = str(file)
        if any(source_file == dep or source_file.startswith(dep + "/") for dep in source_deps):
            covered_source.add(source_file)
            continue
        parent = str(file.parent)
        if file.name == "__init__.py" and any(parent == dep or parent.startswith(dep + "/") for dep in source_deps):
            covered_source.add(source_file)

    all_source = {str(file) for file in SOURCE_ROOT.rglob("*.py")} - GENERATED_SOURCE_FILES
    for path in sorted(all_source - covered_source):
        result.error("Uncovered source files", path)

    init_uncovered = sorted(str(file) for file in SOURCE_ROOT.rglob("__init__.py") if str(file) not in covered_source)
    for path in init_uncovered:
        result.warning("Uncovered __init__.py files", path)


def check_untriggered_tests(config: list[dict[str, Any]], result: CheckResult) -> None:
    test_to_modules = defaultdict(list)
    for module in config:
        module_name = module.get("name", "<unknown>")
        for test_path in module.get("tests", []):
            for expanded in expand_test_path(test_path):
                test_to_modules[pytest_node_file_path(expanded)].append(module_name)
    for test, modules in sorted(test_to_modules.items()):
        if not modules:
            result.warning("Tests without triggering modules", test)


def print_section(kind: str, values: dict[str, list[str]]) -> None:
    print(f"\n{kind}")
    print("=" * 70)
    if not values:
        print("    None")
        return
    for category, items in sorted(values.items()):
        print(f"\n[{category}] {len(items)}")
        for item in sorted(items):
            print(f"    - {item}")


def main() -> int:
    result = CheckResult()
    raw_config = load_config(result)
    if raw_config:
        check_duplicate_modules(raw_config, result)
    config = resolve_config(raw_config, result) if raw_config and not result.has_errors() else []
    if config:
        check_paths(config, result)
        check_e2e_routing(config, result)
        check_test_coverage(config, result)
        check_source_coverage(config, result)
        check_untriggered_tests(config, result)

    print("=" * 70)
    print("TEST CONFIG REVIEW RESULT")
    print("=" * 70)
    print_section("ERRORS", result.errors)
    print_section("WARNINGS", result.warnings)
    return 1 if result.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
