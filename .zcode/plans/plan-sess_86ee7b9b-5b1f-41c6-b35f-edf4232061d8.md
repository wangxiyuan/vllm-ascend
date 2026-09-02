将 e2e-upstream_pr 迁入 test_config.yaml 体系，合并进 run-selected-tests，删除独立 upstream workflow

前提事实（探索确认）：
- upstream 用例（tests/v1/worker/test_mamba_utils.py、tests/lora/test_lora_manager.py）位于 vLLM checkout（./vllm-empty），执行前需向其 tests/conftest.py 注入 import vllm_ascend.patch.platform/worker，依赖 tblib/clang-15 及三个 env。
- select_tests.py 的 curated 流程（--curated，a5 已在用）经 _route_explicit_test_target 路由，但有两道校验会丢弃 vLLM 仓库路径：_is_test_path（只认 tests/ut|tests/e2e）和 path.exists()（在 vllm-ascend 仓库检查）。
- test_config.yaml 的 partition.a2-1 的 runner_label 是 linux-aarch64-a2b3-1，与原 upstream job 用的机器是同一池子。
- run_suite.py/upstream_config.yaml 不再使用（e2e-upstream_singlecard 等其它 suite 仍被 schedule_e2e_upstream_test.yaml 使用，upstream_config.yaml 文件本次保留，只删 e2e-upstream_pr 段）。

改动清单：

1. `.github/workflows/scripts/test_config.yaml`
   - curated_tests 新增 upstream_pr 套件，schema 用 dict 形式标记仓库来源（a5 保持原 list 形式不变）：
     ```yaml
     curated_tests:
       a5:
         - tests/e2e/pull_request/four_card/test_data_parallel_tp2.py
       upstream_pr:
         repo: vllm
         tests:
           - tests/v1/worker/test_mamba_utils.py
           - tests/lora/test_lora_manager.py
     ```
   - estimated_times 新增两个条目（沿用原 upstream_config.yaml 记录的值，各 20 秒）：
     tests/v1/worker/test_mamba_utils.py: 20
     tests/lora/test_lora_manager.py: 20
     这样它们以真实耗时参与 _partition_tests 的动态分组计算，而不是吃 600s 默认值把同分区普通用例的切分挤偏。

2. `.github/workflows/scripts/select_tests.py`
   - _load_curated_tests 兼容两种 schema：list（现状）和 {repo, tests}（repo=vllm）。
   - curated 路由时对 repo=vllm 的 target：跳过 _is_test_path 和 path.exists() 校验，路由到 a2-1 分区（与原 upstream 机器同池），并以 `vllm::` 前缀写入分组（如 `vllm::tests/lora/test_lora_manager.py`），使下游可区分执行方式。
   - upstream target 带着上面配置的 estimated_time 参与 a2-1 分区的正常 load-balanced 切分，自然实现"分到哪个 partition 哪个 partition 跑"。

3. `.github/workflows/scripts/run_selected_tests.sh`
   - 对 `vllm::` 前缀的 target：cd 到 ./vllm-empty（该目录在 _selected_tests.yaml 中已 checkout），首次遇到时做一次性准备（安装 tblib、向 vllm-empty/tests/conftest.py 第 5 行 sed 注入两行 vllm_ascend patch import，设置 PYTORCH_NPU_ALLOC_CONF/VLLM_WORKER_MULTIPROC_METHOD/TORCH_DEVICE_BACKEND_AUTOLOAD），然后以 vllm-empty 为工作目录对该 target 执行 pytest。日志沿用现有 per-target 机制。

4. `.github/workflows/pr_test.yaml`
   - 删除 run-selected-tests-upstream job。
   - select-tests job：仅在 scope-rec（recommended 模式，即 ready-precise 场景）的 select_tests.py 主调用之后，追加一次 `select_tests.py --curated upstream_pr`，把产出的分组与主分组合并（jq/python 合并 JSON 数组）后写入 scope-rec 的 test_groups 输出。scope-all（ready-all）不合并，保持"仅精准测试带 upstream"的现状语义。
   - 合并后若 a2-1 分区没有其它普通用例，会新出现一个 upstream 专属分组——这是必要的（upstream 总需要一台 NPU 机器），但通常 one_card 用例已在 a2-1 池子里，upstream 会搭车执行，不再独占一台。

5. 删除 `.github/workflows/_selected_tests_upstream.yaml`
   - _selected_tests.yaml 无需改动（vllm:: target 由 run_selected_tests.sh 处理）。
   - ci-gate 无需改动（本就未引用 run-selected-tests-upstream）。

6. `.github/workflows/scripts/upstream_config.yaml`
   - 仅删除 e2e-upstream_pr 段（其它 suite 仍被 schedule workflow 使用，文件保留给用户后续处理）。

语义变化（交付时告知）：
- upstream 用例不再独占机器，按分区切分搭车在 a2-1（910B 单卡）分组里与普通用例同机执行。
- 原 ready-precise 且 has_tests=false 时 upstream 仍会跑；合并后随主流程一并跳过。
- upstream 用例会出现在选测清单输出中，与普通用例一起对用户透明。

验证：
- python3 select_tests.py --curated upstream_pr 本地跑通，确认产出 a2-1 分组、vllm:: 前缀、has_tests=true，且 estimated_time 生效。
- python3 select_tests.py --curated a5 回归，确认 a5 路径不受 schema 改动影响。
- --test-list-file 与 --all-tests 模式回归。
- bash -n 校验 run_selected_tests.sh；YAML 语法校验改动的 yaml。
- grep 确认无 _selected_tests_upstream.yaml 残留引用。
- NPU 实跑依赖 CI。