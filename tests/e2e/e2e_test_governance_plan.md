# vLLM Ascend E2E 精准测试治理开发计划

## 背景

`tests/e2e/pull_request` 覆盖 PR E2E 测试，但存在测试组织混乱、覆盖不足、重复覆盖、耗时过长、分层不清等问题。

`.github/workflows/scripts/test_config.yaml` 负责 CI 精准测试映射，但当前规则较初级，存在覆盖不全和冗余触发问题。

## 目标

1. 让 `test_config.yaml` 中“源码变更 -> 测试集合”的映射更精准、可校验、可持续刷新。
2. 让 `tests/e2e/pull_request` 中测试内容更合理、更稳定、更易维护。
3. 优先使用真实运行数据和 coverage 结果，避免依赖人工经验或手写 AST 推断精准测试关系。

## 总体原则

- 先观测，后修改。
- 先生成审计报告，不直接重写 `test_config.yaml`。
- 覆盖映射优先使用 `coverage.py dynamic_context = test_function` 或 `pytest-cov`。
- PR E2E 应短、稳、覆盖核心路径。
- `tests/e2e/pull_request` 本身就是 PR 级 E2E 集合，不再额外维护 tier/owner。
- smoke 测试通过固定子目录表达，例如 `tests/e2e/pull_request/smoke/`。
- 大模型、长序列、性能、多组合矩阵类测试默认迁移出 PR E2E，放到 full/nightly 体系。
- 每个 E2E 测试应有明确的 feature、card、model、hardware 信息。
- 自动化脚本应能持续运行，并逐步接入 CI 守门。

## 推荐新增目录

```text
tests/e2e/tools/
  collect_e2e_profile.py
  collect_e2e_coverage.py
  audit_test_config.py
  check_e2e_metadata.py
  check_test_config.py
  suggest_test_config.py

tests/e2e/reports/
  e2e_profile.json
  e2e_profile.md
  e2e_coverage_map.json
  e2e_quality_audit.md
  test_config_audit.md
```

## Phase 1：测试画像与质量审计

### 1. 新增 E2E 测试画像脚本

文件：`tests/e2e/tools/collect_e2e_profile.py`

目标：扫描 `tests/e2e/pull_request` 下所有 E2E 测试，生成测试画像。

输出：

- `tests/e2e/reports/e2e_profile.json`
- `tests/e2e/reports/e2e_profile.md`

每个测试用例建议记录：

```json
{
  "file": "tests/e2e/pull_request/one_card/test_qwen3_0_6b.py",
  "test": "test_xxx",
  "nodeid": "tests/e2e/pull_request/one_card/test_qwen3_0_6b.py::test_xxx",
  "card": 1,
  "is_310p": false,
  "models": ["Qwen/Qwen3-0.6B"],
  "features": ["dense", "generate"],
  "is_smoke": false,
  "hardware": null,
  "markers": [],
  "skip": false,
  "xfail": false,
  "parametrize": true,
  "uses_vllm_serve": true,
  "uses_llm_offline": false,
  "is_performance_test": false,
  "is_long_sequence_test": false,
  "has_assertion": true,
  "estimated_risk": ["missing_metadata"]
}
```

识别维度：

- 文件路径、pytest nodeid、卡数、是否 310P。
- 模型名称、功能标签、pytest marker。
- skip / xfail / parametrize。
- 是否使用 `vllm serve`、是否 offline `LLM`。
- 是否性能测试、是否长序列测试、是否包含明确断言。
- 是否 smoke、是否缺 metadata、是否疑似大模型、是否疑似慢测试。
- PR CI 适配性：优先使用小参数模型；无小模型时识别是否使用减层、裁剪、Pruning、random/tiny 测试模型。

实现建议：

- 使用 `pytest --collect-only -q` 获取真实 nodeid。
- 可使用 Python AST 做静态信息抽取，但不要用 AST 推断 coverage。
- feature 优先基于 vLLM/vLLM Ascend 的 API 参数、环境变量、配置字段识别，例如 `speculative_config`、`compilation_config`、`additional_config`、`tensor_parallel_size`、`VLLM_ASCEND_*`。

### 2. 新增 E2E 质量审计报告

文件：`tests/e2e/reports/e2e_quality_audit.md`

审计项：

1. 缺 metadata 的测试。
2. 没有明确断言的测试。
3. PR 中疑似大模型测试。
4. PR 中疑似长序列测试。
5. PR 中疑似性能测试。
6. 同模型、同 feature、同卡数的重复测试。
7. 公共 serve/request/assert 逻辑重复。
8. skip/xfail 长期存在的测试。
9. 不在 `test_config.yaml` 中被任何模块触发的测试。
10. 映射到过多模块的高冗余测试。

## Phase 2：真实 coverage 映射采集

### 3. 新增 coverage 采集脚本

文件：`tests/e2e/tools/collect_e2e_coverage.py`

目标：基于 coverage.py dynamic context 收集真实关系：

```text
测试用例 -> vllm_ascend 源码文件
源码文件 -> 测试用例
```

推荐 coverage 配置：

```ini
[run]
branch = True
source = vllm_ascend
dynamic_context = test_function
```

输出：`tests/e2e/reports/e2e_coverage_map.json`

运行示例：

```bash
python tests/e2e/tools/collect_e2e_coverage.py --tests tests/e2e/pull_request/one_card/test_qwen3_0_6b.py
python tests/e2e/tools/collect_e2e_coverage.py --tests tests/e2e/pull_request/one_card/test_qwen3_0_6b.py::test_dense_default_full_and_piecewise_graph
python tests/e2e/tools/collect_e2e_coverage.py --append --tests tests/e2e/pull_request/two_card -- -k prefix
```

临时 PR CI 采集方式：给 PR 添加 `smart-e2e` 和 `ready` label 后，PR E2E 会使用全量 configured test scope，并上传 `phase2-e2e-coverage-*` artifacts。

要求：

- 支持传入单个测试文件、目录或 nodeid。
- 支持合并历史 coverage map。
- 支持 Python subprocess coverage startup，用当前 pytest nodeid 作为子进程静态 context。
- 只关注 `vllm_ascend/` 下源码。
- 不自研 AST coverage。
- 如果环境无法运行 E2E，应清晰报错，不破坏已有文件。

## Phase 3：审计 test_config.yaml

### 4. 新增 test_config 审计脚本

文件：`tests/e2e/tools/audit_test_config.py`

输入：

- `.github/workflows/scripts/test_config.yaml`
- `tests/e2e/reports/e2e_coverage_map.json`
- `tests/e2e/reports/e2e_profile.json`

输出：`tests/e2e/reports/test_config_audit.md`

审计项：

1. `test_config.yaml` 引用不存在的测试文件。
2. `source_file_dependencies` 路径不存在。
3. `base` 引用不存在或存在循环继承。
4. module name 重复。
5. 某些 E2E 测试无人触发。
6. 某些源码文件没有任何 E2E 覆盖。
7. 某些源码文件有 coverage 结果，但 `test_config.yaml` 未配置对应测试。
8. 某些模块配置过宽，触发大量不相关测试。
9. 某些测试被过多模块复用，疑似 fallback 测试。
10. `tests: []` 模块是否需要人工确认。
11. 目录级配置是否可收敛为文件级或 nodeid 级。

## Phase 4：建立 E2E metadata 与 smoke 目录规范

### 5. 推荐 metadata 规范

`tests/e2e/pull_request` 本身就是 PR 级 E2E 测试集合，不再维护 `tier` 和 `owner`。smoke 测试通过固定子目录表达：

```text
tests/e2e/pull_request/smoke/
```

推荐使用 pytest marker 标注必要维度：

```python
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.feature("spec_decode"),
    pytest.mark.cards(1),
    pytest.mark.model("Qwen/Qwen3-0.6B"),
]
```

推荐字段：

| 字段 | 必填 | 示例 |
|---|---|---|
| feature | 是 | attention / spec_decode / lora / quantization |
| cards | 是 | 1 / 2 / 4 |
| model | 是 | Qwen/Qwen3-0.6B |
| hardware | 310P 必填 | 310p / 910b / 910c |
| duration | 可选 | short / medium / long |

目录规则：

| 目录 | 定义 |
|---|---|
| `tests/e2e/pull_request/smoke/` | 每次 PR 必跑，少量核心链路，小模型，短耗时，高稳定 |
| `tests/e2e/pull_request/{one_card,two_card,four_card}/` | 按源码模块精准触发的 PR E2E |
| `tests/e2e/full/` 或 nightly workflow | 大模型、长序列、多组合、低频功能、性能测试 |

### 6. 新增 metadata 检查脚本

文件：`tests/e2e/tools/check_e2e_metadata.py`

输出：

- `tests/e2e/reports/e2e_metadata_check.json`
- `tests/e2e/reports/e2e_metadata_check.md`

检查项：

1. 是否有 feature。
2. 是否有 cards。
3. 是否有 model。
4. `one_card/two_card/four_card` 路径与 cards marker 是否一致。
5. 310P 文件是否缺 hardware marker。
6. `tests/e2e/pull_request/smoke/` 下测试是否满足短耗时、小模型、稳定断言要求。

初期策略：

- CI 中只 warning，不阻塞。
- 新增测试必须满足 metadata。
- 存量测试逐步补齐。

## Phase 5：test_config 自动校验与刷新

### 7. 优化现有 test_config 校验脚本

文件：`.github/workflows/scripts/coverage.py`

现状：该脚本已在 PR CI 中通过 `Validate test coverage config` 步骤运行，当前可检查：

1. `test_config.yaml` 中 `tests` 路径是否存在。
2. `tests/e2e/pull_request` 下测试文件是否被配置覆盖。
3. `tests/ut` 下测试文件是否被配置覆盖。
4. `vllm_ascend/` 源码文件是否被 `source_file_dependencies` 覆盖。

目标：不要新增 `tests/e2e/tools/check_test_config.py`，而是重构/增强现有 `.github/workflows/scripts/coverage.py`，作为 Phase 5 基础校验入口。

增强项：

1. YAML 可解析，并输出清晰错误。
2. 所有 `tests` 路径存在。
3. 所有 `source_file_dependencies` 路径存在。
4. `base` 引用存在。
5. 没有循环继承。
6. 没有重复 module name。
7. `skip_tests` 路径存在。
8. 目录配置展开后不为空。
9. E2E 路径符合 runner 路由规则。
10. 发现明显无人触发测试时 warning。

实现要求：

- 复用 `.github/workflows/scripts/select_tests.py` 中已有的继承解析、路径展开、skip 过滤和 runner routing helper，避免两套逻辑漂移。
- 保持当前 CI 入口不变：`python3 .github/workflows/scripts/coverage.py`。
- Phase 5 基础校验不依赖 Phase 2 coverage 结果；后续自动建议/刷新仍依赖 Phase 2/3。

### 8. 后续可选：生成 test_config 建议

文件：`tests/e2e/tools/suggest_test_config.py`

目标：基于 coverage map 生成建议 diff，而不是直接覆盖配置。

## Phase 6：E2E 重构策略

第一批只做低风险重构：

1. 补 metadata/marker。
2. 抽公共 fixture/utils。
3. 合并明显重复的参数化测试。
4. 将性能测试迁移出 `pull_request`，进入 full/nightly 体系。
5. 将长序列测试迁移出 `pull_request`，进入 full/nightly 体系。
6. 将大模型低频组合从 PR 精准测试中移出。
7. 修复无效 skip/xfail。
8. 删除或迁移无人触发且无覆盖价值的测试前，必须有审计证据。

第一批不建议做：

- 大规模删除测试。
- 大规模重写 `test_config.yaml`。
- 改动模型行为。
- 同时修改 CI runner 路由。

## AI Agent 任务拆分

### Agent A1：E2E 测试画像 Agent

```text
你是 vLLM Ascend E2E 测试画像 Agent。

工作目录：/home/wxy/vllm-ascend

目标：
1. 扫描 tests/e2e/pull_request 下所有 E2E 测试。
2. 生成每个测试文件/测试函数的画像。
3. 输出 e2e_profile.json、e2e_profile.md、e2e_quality_audit.md。

要求：
- Phase 1 只做观测，不修改现有测试文件。
- 可以新增 tests/e2e/tools/collect_e2e_profile.py。
- 可以使用 pytest --collect-only 获取 nodeid。
- 可以使用 AST 做静态信息抽取，但不要用 AST 推断 coverage。
- 每个审计问题必须给出文件路径和测试函数名。
- 完成后运行 ruff check 对新增脚本做校验。
```

### Agent A2：Coverage 映射 Agent

```text
你是 vLLM Ascend 精准测试 coverage 映射 Agent。

工作目录：/home/wxy/vllm-ascend

目标：
1. 基于 coverage.py dynamic_context = test_function 设计并实现 E2E coverage 采集能力。
2. 新增 tests/e2e/tools/collect_e2e_coverage.py。
3. 支持运行指定 pytest 测试文件、目录或 nodeid。
4. 输出 tests/e2e/reports/e2e_coverage_map.json。
5. 只统计 vllm_ascend/ 下源码文件。

要求：
- 优先使用 coverage.py / pytest-cov，不要自研 AST coverage。
- 支持 --tests、--output、--append 参数。
- 如果环境不能运行 E2E，脚本应清晰报错。
- 不修改现有测试文件。
- 完成后运行 ruff check 对新增脚本做校验。
```

### Agent A3：test_config 审计 Agent

```text
你是 vLLM Ascend test_config.yaml 审计 Agent。

工作目录：/home/wxy/vllm-ascend

目标：
1. 新增 tests/e2e/tools/audit_test_config.py。
2. 读取 .github/workflows/scripts/test_config.yaml、e2e_coverage_map.json、e2e_profile.json。
3. 输出 tests/e2e/reports/test_config_audit.md。
4. 审计引用不存在的测试路径、source 路径不存在、base 引用不存在、base 循环继承、module name 重复、E2E 测试无人触发、源码有 coverage 但 test_config 未配置、配置过宽、tests: [] 模块等问题。

要求：
- 兼容 test_config.yaml 中现有 base、skip_tests、exclude_source_file_dependencies 结构。
- 路径统一使用 repo 相对路径。
- 不直接修改 test_config.yaml，只输出报告。
- 完成后运行 ruff check 对新增脚本做校验。
```

### Agent A4：metadata 检查 Agent

```text
你是 vLLM Ascend E2E metadata 规范 Agent。

工作目录：/home/wxy/vllm-ascend

目标：
1. 设计 E2E metadata 规范：feature、cards、model、hardware。
2. 新增 tests/e2e/tools/check_e2e_metadata.py。
3. 检查 tests/e2e/pull_request 下所有测试是否符合 metadata 规范。
4. 输出终端报告，并支持 --format json。
5. 初期只 warning，不修改测试文件。

要求：
- 不引入 tier/owner。
- smoke 仅通过 tests/e2e/pull_request/smoke/ 目录表达。
- 检查路径卡数与 cards marker 是否一致。
- 检查 310P 测试是否有 hardware 信息。
- 完成后运行 ruff check 对新增脚本做校验。
```

## 推荐推进顺序

1. 先做 Agent A1，拿到测试画像和质量审计。
2. 再做 Agent A2，拿到真实 coverage 映射。
3. 再做 Agent A3，对 `test_config.yaml` 做审计。
4. 再做 Agent A4，建立 metadata 守门。
5. 最后基于报告分批重构 E2E 测试和精准测试配置。
