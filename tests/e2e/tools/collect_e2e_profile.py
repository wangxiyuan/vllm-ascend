import ast
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
E2E_PR_ROOT = REPO_ROOT / "tests" / "e2e" / "pull_request"
REPORTS_DIR = REPO_ROOT / "tests" / "e2e" / "reports"
MODEL_KEYS = ("model", "model_name", "draft_model", "eagle_model")
RUNNERS = {"VllmRunner", "RemoteOpenAIServer"}
ASSERT_CALLS = {"compare_logprobs", "check_outputs", "check_models_equal", "assert_outputs_equal"}
LARGE_MODELS = ("30b", "32b", "35b", "70b", "72b", "80b", "110b", "235b", "deepseek-v", "gpt-oss")
REDUCED_MODEL_HINTS = ("pruning", "lite", "random", "tiny", "small", "reduced", "dummy")
SMALL_MODEL_MAX_B = 8
MEDIUM_MODEL_MAX_B = 32
FEATURE_SIGNALS = {
    "aclgraph": {
        "cudagraph_mode",
        "cudagraph_capture_sizes",
        "max_cudagraph_capture_size",
        "enable_npugraph_ex",
        "enable_static_kernel",
        "xlite_graph_config",
    },
    "ascend_fusion": {
        "ascend_compilation_config",
        "ascend_fusion_config",
        "fuse_norm_quant",
        "fuse_qknorm_rope",
        "fuse_allreduce_rms",
        "fuse_muls_add",
        "fusion_ops_gmmswigluquant",
    },
    "chunked_prefill": {"enable_chunked_prefill", "long_prefill_token_threshold"},
    "context_parallel": {"prefill_context_parallel_size", "decode_context_parallel_size"},
    "data_parallel": {"data_parallel_size", "data_parallel_backend", "data_parallel_rank"},
    "distributed": {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "distributed_executor_backend",
        "nnodes",
        "node_rank",
    },
    "eplb": {"enable_eplb", "eplb_config", "dynamic_eplb", "DYNAMIC_EPLB"},
    "expert_parallel": {"enable_expert_parallel", "enable_ep_weight_filter", "expert_placement_strategy"},
    "flashcomm": {
        "enable_flashcomm1",
        "enable_flashcomm2_parallel_size",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1",
        "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE",
    },
    "guided_decoding": {"guided_decoding_backend", "guided_json", "guided_regex", "guided_choice"},
    "kv_offload": {"cpu_offload_gb", "swap_space", "enable_kv_nz", "kv_transfer_config", "kv_connector"},
    "lora": {"enable_lora", "lora_modules", "fully_sharded_loras", "max_loras", "olora_tensor_parallel_size"},
    "multimodal": {
        "limit_mm_per_prompt",
        "mm_processor_kwargs",
        "compile_mm_encoder",
        "cudagraph_mm_encoder",
        "encoder_cudagraph_token_budgets",
        "image",
        "images",
        "video",
        "audio",
    },
    "pooling": {"runner_type", "pooling", "score", "embed", "classify", "task"},
    "prefix_caching": {"enable_prefix_caching", "prefix_caching_hash_algo"},
    "quantization": {"quantization", "quant_config", "weight_nz_mode", "VLLM_ASCEND_ENABLE_NZ"},
    "sampling": {"enable_reduce_sample", "logprobs", "top_k", "top_p", "temperature"},
    "scheduler": {"async_scheduling", "enable_balance_scheduling", "SLO_limits_for_dynamic_batch"},
    "spec_decode": {"speculative_config", "num_speculative_tokens", "draft_tensor_parallel_size"},
    "weight_prefetch": {"weight_prefetch_config", "prefetch_ratio"},
}

METHOD_FEATURE_VALUES = {
    "spec_decode": {"eagle", "eagle3", "mtp", "ngram", "suffix", "dflash", "draft", "extract_hidden_states"},
    "aclgraph": {"FULL", "FULL_DECODE_ONLY", "PIECEWISE", "FULL_AND_PIECEWISE"},
}

MODEL_FEATURE_PATTERNS = {
    "embedding": ("embedding", "bge", "e5"),
    "classification": ("classification", "classify"),
    "moe": ("moe", "a3b", "deepseek-v", "gpt-oss"),
    "multimodal": ("-vl", "_vl", "vl-", "vlm", "vision", "audio", "whisper", "minicpm", "ocr"),
    "quantization": ("w8a8", "w4a8", "mxfp8", "int8", "quant", "pruning"),
}

PATH_FEATURE_PATTERNS = {
    "attention": ("attention", "fa3", "sfa", "dsa", "mla"),
    "batch_invariant": ("batch_invariant",),
    "compile": ("compile", "fusion", "graphex"),
    "external_launcher": ("external_launcher",),
    "long_sequence": ("long_sequence",),
    "performance": ("performance", "profiling", "benchmark"),
    "pooling": ("pooling", "scoring"),
}


@dataclass
class TestProfile:
    file: str
    test: str
    nodeid: str
    card: int | None
    is_310p: bool
    is_smoke: bool
    models: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    hardware: str | None = None
    feature_signals: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    skip: bool = False
    xfail: bool = False
    parametrize: bool = False
    uses_vllm_serve: bool = False
    uses_llm_offline: bool = False
    uses_subprocess: bool = False
    uses_vllm_runner: bool = False
    is_performance_test: bool = False
    is_long_sequence_test: bool = False
    has_assertion: bool = False
    model_size_risk: str = "unknown"
    uses_reduced_layers: bool = False
    ci_suitability: str = "unknown"
    ci_suitability_reasons: list[str] = field(default_factory=list)
    estimated_risk: list[str] = field(default_factory=list)


def literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (SyntaxError, ValueError):
        return node.id if isinstance(node, ast.Name) else None


def string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result = []
        for key, value_item in value.items():
            if isinstance(key, str) and any(model_key in key.lower() for model_key in MODEL_KEYS):
                result.extend(string_values(value_item))
        return result
    if isinstance(value, (list, tuple, set)):
        return [item for value_item in value for item in string_values(value_item)]
    return []


def plain_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for value_item in value.values() for item in plain_string_values(value_item)]
    if isinstance(value, (list, tuple, set)):
        return [item for value_item in value for item in plain_string_values(value_item)]
    return []


def looks_like_model(value: str) -> bool:
    lower = value.lower()
    names = ("qwen", "deepseek", "llama", "minicpm", "gpt", "bge", "whisper", "hunyuan", "baai")
    if value in {"model", "main", "spec", "cur_case"}:
        return False
    if " " in value or "<" in value or value.endswith((".png", ".jpg", ".jpeg")):
        return False
    return "/" in value or lower.startswith("e5") or "e5-" in lower or any(name in lower for name in names)


def model_strings_from_node(node: ast.AST) -> list[str]:
    values: list[str] = []
    value = None if isinstance(node, ast.Name) else literal(node)
    values.extend(item for item in plain_string_values(value) if looks_like_model(item))
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            for arg in child.args[:1]:
                values.extend(item for item in plain_string_values(literal(arg)) if looks_like_model(item))
            for keyword in child.keywords:
                if keyword.arg and any(key in keyword.arg.lower() for key in MODEL_KEYS):
                    values.extend(
                        item for item in plain_string_values(literal(keyword.value)) if looks_like_model(item)
                    )
    return values


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def collect_assigned_models(tree: ast.AST) -> dict[str, list[str]]:
    result = defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = literal(node.value)
        field_values = [item for item in string_values(value) if looks_like_model(item)]
        plain_values = model_strings_from_node(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if field_values:
                result[target.id].extend(field_values)
            elif target.id.isupper() or any(key in target.id.lower() for key in MODEL_KEYS):
                result[target.id].extend(plain_values)
    return result


def extract_local_models(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[str]]:
    result = defaultdict(list)
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        value = literal(node.value)
        field_values = [item for item in string_values(value) if looks_like_model(item)]
        plain_values = model_strings_from_node(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if field_values:
                result[target.id].extend(field_values)
            elif any(key in target.id.lower() for key in MODEL_KEYS):
                result[target.id].extend(plain_values)
    return result


def extract_models(function: ast.FunctionDef | ast.AsyncFunctionDef, known: dict[str, list[str]]) -> list[str]:
    values = []
    local_models = extract_local_models(function)
    model_arg_names = {
        arg.arg
        for arg in function.args.args
        if arg.arg.lower().endswith("model_name") or arg.arg.lower() in {"vl_model_name"}
    }
    values.extend(sorted(model_arg_names))
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            values.extend(local_models.get(node.id, []))
            values.extend(known.get(node.id, []))
        if isinstance(node, ast.Call):
            call = call_name(node.func)
            if call in RUNNERS:
                for arg in node.args[:1]:
                    if isinstance(arg, ast.Name):
                        values.extend(local_models.get(arg.id, []))
                        values.extend(known.get(arg.id, []))
                    if not isinstance(arg, ast.Name):
                        values.extend(value for value in plain_string_values(literal(arg)) if looks_like_model(value))
            for keyword in node.keywords:
                if keyword.arg and any(key in keyword.arg.lower() for key in MODEL_KEYS):
                    if isinstance(keyword.value, ast.Name):
                        values.extend(local_models.get(keyword.value.id, []))
                        values.extend(known.get(keyword.value.id, []))
                    if not isinstance(keyword.value, ast.Name):
                        values.extend(
                            value for value in plain_string_values(literal(keyword.value)) if looks_like_model(value)
                        )
    return sorted(set(values))


def detect_card(path: str) -> int | None:
    parts = path.split("/")
    if "one_card" in parts:
        return 1
    if "two_card" in parts:
        return 2
    if "four_card" in parts:
        return 4
    return None


def model_param_billions(model: str) -> float | None:
    text = model.lower()
    for index, char in enumerate(text):
        if char != "b":
            continue
        start = index - 1
        while start >= 0 and (text[start].isdigit() or text[start] == "."):
            start -= 1
        value = text[start + 1 : index]
        if value:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def has_reduced_hint(text: str) -> bool:
    return any(hint in text.lower() for hint in REDUCED_MODEL_HINTS)


def uses_reduced_layers(models: list[str], source: str) -> bool:
    text = "\n".join(models + [source]).lower()
    return has_reduced_hint(text) or "num_hidden_layers" in text


def is_large_model_name(model: str) -> bool:
    size = model_param_billions(model)
    if size is not None:
        return size > MEDIUM_MODEL_MAX_B
    model_lower = model.lower()
    return "deepseek-v3" in model_lower or "deepseek-v4" in model_lower or "gpt-oss" in model_lower


def is_reduced_model_name(model: str) -> bool:
    model_lower = model.lower()
    return has_reduced_hint(model_lower) or "lite" in model_lower


def has_large_model(models: list[str]) -> bool:
    return any(is_large_model_name(model) for model in models)


def has_large_model_without_reduction(models: list[str], source: str) -> bool:
    if "num_hidden_layers" in source.lower():
        return False
    return any(is_large_model_name(model) and not is_reduced_model_name(model) for model in models)


def model_size_risk(models: list[str]) -> str:
    sizes = [size for model in models if (size := model_param_billions(model)) is not None]
    if not sizes:
        return "unknown"
    max_size = max(sizes)
    if max_size <= SMALL_MODEL_MAX_B:
        return "small"
    if max_size <= MEDIUM_MODEL_MAX_B:
        return "medium"
    return "large"


def ci_suitability(
    profile: TestProfile,
    source: str,
) -> tuple[str, list[str]]:
    reasons = []
    if profile.is_performance_test:
        reasons.append("performance_test")
    if profile.is_long_sequence_test:
        reasons.append("long_sequence")
    if profile.card == 4 and profile.model_size_risk == "large":
        reasons.append("four_card_large_model")
    if has_large_model_without_reduction(profile.models, source):
        reasons.append("large_model_without_reduction")
    if "235b" in " ".join(profile.models).lower():
        reasons.append("larger_than_minimal_pr_model")
    if "deepseek-v3" in " ".join(profile.models).lower() and not profile.uses_reduced_layers:
        reasons.append("deepseek_v3_requires_reduction")
    if "deepseek-v4" in " ".join(profile.models).lower() and not profile.uses_reduced_layers:
        reasons.append("deepseek_v4_requires_reduction")
    if "Qwen3-235B" in source:
        reasons.append("qwen3_moe_235b_should_use_30b")
    if reasons:
        severe = {"performance_test", "long_sequence", "large_model_without_reduction", "qwen3_moe_235b_should_use_30b"}
        return ("nightly_only" if severe & set(reasons) else "pr_risky"), sorted(set(reasons))
    return "pr_ok", []


def collect_signal_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    signals = set()
    for node in ast.walk(function):
        if isinstance(node, ast.keyword) and node.arg:
            signals.add(node.arg)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                value = literal(key) if key is not None else None
                if isinstance(value, str):
                    signals.add(value)
        elif isinstance(node, ast.Name):
            signals.add(node.id)
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name:
                signals.add(name)
    return signals


def detect_features(
    path: str,
    test: str,
    source: str,
    models: list[str],
    signals: set[str],
) -> tuple[list[str], list[str]]:
    features: list[str] = []
    evidence: list[str] = []
    lowered_signals = {signal.lower(): signal for signal in signals}
    for feature, feature_signals in FEATURE_SIGNALS.items():
        matched_signals = {signal.lower() for signal in feature_signals} & set(lowered_signals)
        if matched_signals:
            features.append(feature)
            evidence.extend(lowered_signals[item] for item in matched_signals)
    for feature, values in METHOD_FEATURE_VALUES.items():
        matched_values = [value for value in values if value.lower() in source.lower()]
        if matched_values:
            features.append(feature)
            evidence.extend(matched_values)
    model_text = " ".join(models).lower()
    for feature, patterns in MODEL_FEATURE_PATTERNS.items():
        matched_patterns = [pattern for pattern in patterns if pattern in model_text]
        if matched_patterns:
            features.append(feature)
            evidence.extend(matched_patterns)
    path_text = f"{path}\n{test}".lower()
    for feature, patterns in PATH_FEATURE_PATTERNS.items():
        matched_path_patterns = [pattern for pattern in patterns if pattern in path_text]
        if matched_path_patterns:
            features.append(feature)
            evidence.extend(matched_path_patterns)
    if not features and models:
        features.append("model_generation")
        evidence.extend(models[:3])
    return sorted(set(features)), sorted(set(evidence))


def collect_markers(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return sorted({name for item in function.decorator_list if (name := dotted_name(item))})


def uses_name(function: ast.FunctionDef | ast.AsyncFunctionDef, names: set[str]) -> bool:
    return any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(function))


def has_assertion(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call) and call_name(node.func) in ASSERT_CALLS:
            return True
    return False


def detect_hardware(path: str, source: str) -> str | None:
    text = f"{path}\n{source}".lower()
    for item in ("310p", "910b", "910c"):
        if item in text:
            return item
    return None


def estimate_risks(profile: TestProfile) -> list[str]:
    risks = []
    if not profile.features:
        risks.append("missing_feature")
    if not profile.models:
        risks.append("missing_model")
    if profile.card is None:
        risks.append("missing_card")
    if profile.is_310p and profile.hardware != "310p":
        risks.append("missing_310p_hardware")
    if not profile.has_assertion:
        risks.append("no_explicit_assertion")
    if profile.is_performance_test:
        risks.append("performance_test_in_pr")
    if profile.is_long_sequence_test:
        risks.append("long_sequence_test_in_pr")
    if has_large_model(profile.models):
        risks.append("large_model_in_pr")
    if profile.ci_suitability == "nightly_only":
        risks.append("nightly_only_in_pr")
    elif profile.ci_suitability == "pr_risky":
        risks.append("pr_risky_model")
    if "large_model_without_reduction" in profile.ci_suitability_reasons:
        risks.append("large_model_without_reduction")
    if "qwen3_moe_235b_should_use_30b" in profile.ci_suitability_reasons:
        risks.append("qwen3_moe_235b_should_use_30b")
    if profile.skip or profile.xfail:
        risks.append("skipped_or_xfailed")
    return sorted(set(risks))


def profile_function(
    path: Path,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    known: dict[str, list[str]],
) -> TestProfile:
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    source = ast.get_source_segment(text, function) or ""
    markers = collect_markers(function)
    models = extract_models(function, known)
    signals = collect_signal_names(function)
    features, feature_signals = detect_features(rel_path, function.name, source, models, signals)
    reduced_layers = uses_reduced_layers(models, source)
    size_risk = model_size_risk(models)
    profile = TestProfile(
        file=rel_path,
        test=function.name,
        nodeid=f"{rel_path}::{function.name}",
        card=detect_card(rel_path),
        is_310p="310p" in rel_path.lower(),
        is_smoke="/smoke/" in rel_path,
        models=models,
        features=features,
        hardware=detect_hardware(rel_path, source),
        feature_signals=feature_signals,
        markers=markers,
        skip=any("skip" in marker for marker in markers),
        xfail=any("xfail" in marker for marker in markers),
        parametrize="pytest.mark.parametrize" in markers,
        uses_vllm_serve=uses_name(function, {"RemoteOpenAIServer"}),
        uses_llm_offline=uses_name(function, {"LLM"}),
        uses_subprocess=uses_name(function, {"subprocess"}),
        uses_vllm_runner=uses_name(function, RUNNERS),
        is_performance_test="performance" in features or "profiling" in rel_path.lower(),
        is_long_sequence_test="long_sequence" in features or "long_sequence" in rel_path.lower(),
        has_assertion=has_assertion(function),
        model_size_risk=size_risk,
        uses_reduced_layers=reduced_layers,
    )
    suitability, suitability_reasons = ci_suitability(profile, source)
    profile.ci_suitability = suitability
    profile.ci_suitability_reasons = suitability_reasons
    profile.estimated_risk = estimate_risks(profile)
    return profile


def collect_profiles() -> list[TestProfile]:
    profiles = []
    for path in sorted(E2E_PR_ROOT.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        known = collect_assigned_models(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                profiles.append(profile_function(path, node, text, known))
    return sorted(profiles, key=lambda item: (item.file, item.test))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_profile_md(profiles: list[TestProfile]) -> None:
    rows = [
        [
            p.file,
            p.test,
            p.card or "",
            ", ".join(p.models[:3]),
            ", ".join(p.features),
            p.model_size_risk,
            "yes" if p.uses_reduced_layers else "no",
            p.ci_suitability,
            ", ".join(p.ci_suitability_reasons[:3]),
            ", ".join(p.feature_signals[:8]),
            ", ".join(p.estimated_risk),
        ]
        for p in profiles
    ]
    content = "\n".join(
        [
            "# E2E Profile",
            "",
            f"Total tests: {len(profiles)}",
            "",
            table(
                [
                    "File",
                    "Test",
                    "Cards",
                    "Models",
                    "Features",
                    "Model size",
                    "Reduced",
                    "CI suitability",
                    "CI reasons",
                    "Signals",
                    "Risks",
                ],
                rows,
            ),
            "",
        ]
    )
    (REPORTS_DIR / "e2e_profile.md").write_text(content, encoding="utf-8")


def write_quality_audit(profiles: list[TestProfile]) -> None:
    groups = defaultdict(list)
    duplicates = defaultdict(list)
    for profile in profiles:
        for risk in profile.estimated_risk:
            groups[risk].append(profile)
        if profile.models and profile.features:
            duplicates[(profile.card, tuple(profile.features), tuple(profile.models))].append(profile)
    summary_rows = [[risk, len(items)] for risk, items in sorted(groups.items(), key=lambda item: item[0])]
    sections = [
        "# E2E Quality Audit",
        "",
        "## Summary",
        "",
        table(["Risk", "Count"], summary_rows),
        "",
        "## Recommended first actions",
        "",
        "1. Move `long_sequence_test_in_pr` and `performance_test_in_pr` cases out of PR E2E.",
        "2. Replace oversized PR models with the smallest model that covers the same feature.",
        "3. For models without small variants, use reduced-layer/pruned/random-weight variants in PR.",
        "4. Review `large_model_without_reduction` and `qwen3_moe_235b_should_use_30b` first.",
        "5. Fix or remove long-lived `skipped_or_xfailed` cases.",
        "6. Add explicit assertions or common helper assertions for `no_explicit_assertion` cases.",
        "",
    ]
    for risk, items in sorted(groups.items()):
        rows = [[item.nodeid, item.file, ", ".join(item.features), ", ".join(item.models[:3])] for item in items]
        sections.extend([f"## {risk}", "", table(["Test", "File", "Features", "Models"], rows), ""])
    dup_rows = []
    for items in duplicates.values():
        if len(items) > 1:
            dup_rows.append(
                [
                    len(items),
                    items[0].card or "",
                    ", ".join(items[0].features),
                    "<br>".join(item.nodeid for item in items),
                ]
            )
    if dup_rows:
        sections.extend(
            ["## possible_duplicate_groups", "", table(["Count", "Cards", "Features", "Tests"], dup_rows), ""]
        )
    (REPORTS_DIR / "e2e_quality_audit.md").write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    profiles = collect_profiles()
    profile_json = json.dumps([asdict(item) for item in profiles], indent=2, ensure_ascii=False) + "\n"
    (REPORTS_DIR / "e2e_profile.json").write_text(profile_json, encoding="utf-8")
    write_profile_md(profiles)
    write_quality_audit(profiles)
    print(f"Wrote {len(profiles)} test profiles to {REPORTS_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
