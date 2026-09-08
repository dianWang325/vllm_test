# vllm_test

这是一个面向 vLLM Ascend 的配置驱动测试框架：负责服务启动与健康检查、测试数据准备、AISBench 性能或精度请求，以及结构化报告生成。支持非 PD、单机 PD 和双机 PD；双机 Decode 标记为 `external: true`，由对应主机独立启动和停止。

框架遇到 case 或服务生命周期异常后停止，不自动重跑 case。请求级 `retries` 会传给 AISBench，当前性能和精度配置均为 `3`；这与框架级重跑是两回事。

## 目录

```text
vllm_test/
├── vtest
├── scripts/                    # 功能层与命令行接口
├── configs/                    # server、model、case、suite 与报告配置
├── aisbench_workspace/
│   ├── configs/models/vllm_api/vllm_test.py
│   └── datasets/
│       ├── formal/             # AISBench SyntheticDataset 生成的数据
│       ├── warmup/             # aisbench_auto_tools_prefix 生成的数据
│       ├── gsm8k/test.jsonl
│       └── gpqa/gpqa_diamond.csv
├── runs/                       # 扁平运行产物
└── tests/test_pd_configuration.py # 配置、拓扑、预热和请求空闲检查测试
```

AISBench 以环境中的 `ais-bench-benchmark` 包提供，项目内不创建名为 `ais_bench` 的 Python 包。手动预热生成器由 `configs/data.yaml` 中显式的 `tool_path` 指向外部 `aisbench_auto_tools_prefix` 仓库。

## 使用

依赖安装在 `wd_test0825` 容器中，因此应进入该容器运行：

```bash
docker exec -it -w /home/w00985415/vllm_test wd_test0825 bash
./vtest list
./vtest runtime-info --run-type performance
./vtest run case baseline_smoke
./vtest run suite baseline_performance
./vtest run suite prefill_variable_performance
./vtest run suite gpqa_deepseek_v4_flash_accuracy
```

以上是按需选择的独立运行示例。真实服务测试需要 Linux/NPU 环境、可用模型路径以及 vLLM、vllm-ascend 和 AISBench 依赖。

## 配置职责与覆盖

编辑 `configs/*.yaml` 即可增加 model、profile、case 或 suite。每项均可填写 `description`；该字段只用于说明和 `selected.yaml`，不会参与服务、数据或请求参数。服务配置按 `server defaults < server profile < model < case overrides < suite case_overrides` 递归合并，列表和标量直接替换。

| 配置文件 | 职责 |
| --- | --- |
| `cases.yaml` | 定义单个测试；顶部注释表逐项列出当前 25 个 case（19 个性能、6 个精度）。 |
| `suites.yaml` | 按顺序组合 case，并提供 `case_overrides`；顶部注释表列出当前 13 个 suite。 |
| `server.yaml` | 定义非 PD/PD 公共服务结构、参数、环境变量和生命周期。 |
| `model.yaml` | 定义模型标识、默认路径及模型专用参数。 |
| `data.yaml` | 定义性能正式数据和独立手动预热数据。 |
| `bench.yaml` | 定义性能请求参数：`concurrency_4`、`concurrency_8`、`concurrency_128`，以及独立 `warmup`。 |
| `accuracy.yaml` | 定义精度数据集、正式请求并发、最大输出和生成参数。 |
| `report.yaml` | 定义结构化报告必需指标。 |

精度 case 不配置 `bench`，正式请求只读取 `accuracy`（包括 `overrides.accuracy`）。当前精度并发为 128、最大输出为 32768 token；`temperature: 1`、`top_p: 0.95`、`ignore_eos: false`，启用 thinking。`bench.yaml` 中的 `concurrency_128` 不会自动影响精度测试。精度 case 若配置 `warmup`，仅该独立预热阶段使用 `bench.yaml` 的 `warmup` profile，默认并发为 1。

服务 profile 为 `non_pd_baseline`、`non_pd_cpp`、`non_pd_srf`、`non_pd_cpp_srf`、`pd` 和 `pd_two_host`。非 PD profile 共用默认 8 卡（0–7）、TP=4、PP=2；CPP 默认 `smooth_factor: 1.0`，SRF 默认 `threshold: 40960`、`long_max_wait_ms: 2000`。`prefill_variable_performance` 直接复用这四种 profile，不额外覆盖 CPP 参数。

单机与双机 PD 的具体卡号、并行布局、主机地址和网络参数放在对应 suite 的 `case_overrides`；运行单个 case 不会加载 suite 覆盖。跨机角色启动也应使用相同 `--suite`，使两端得到一致的部署配置。

`configs/server.yaml` 中的 `arguments` 和 `environment` 会直接传给 vLLM 服务。`arguments` 的键是完整参数名：值为 `null` 或 `true` 时只输出参数名，`false` 时不输出该参数，字符串或数字作为下一个命令行参数，字典或列表编码为紧凑 JSON。`environment` 的值必须是字符串或数字；`unset_environment` 用于明确删除继承的环境变量。普通服务使用 `--host` 和 `--port` 进行健康检查和请求；跨主机 PD 角色可额外设置 `endpoint_host` 作为其他节点实际访问的地址，并用 `external: true` 表示该角色仅做健康检查、不由当前 vtest 进程启动或停止。基础 profile 不传递 `--additional-config`，CPP 和 SRF 只在各自 profile 中声明所需的调度配置。

## 测试场景与数据

| 场景 | 当前入口与组成 |
| --- | --- |
| 冒烟 | `baseline_smoke`、`cpp_smoke`、`srf_smoke`、`cpp_srf_smoke` 四个 case；均使用 `srf_mixed`、8 并发。单独执行时继承默认 Qwen 模型。 |
| 基线综合验证 | `baseline_smoke` suite 组合冒烟、GSM8K 和 GPQA，并将冒烟模型覆盖为 Flash；这是性能/精度混合 suite。 |
| 非 PD Prefill 定长/变长 | `baseline_performance` 运行 baseline 定长与变长；`cpp_validation` 运行 CPP 定长与变长；`srf_validation` 和 `cpp_srf_validation` 各运行一个变长 case。三个 validation suite 当前不包含冒烟。 |
| 非 PD Prefill 变长对比 | `prefill_variable_performance` 按 baseline → CPP → SRF → CPP+SRF 运行 4 个 case，均为 4 并发、1 token 输出，生成 comparison。 |
| 精度 | `baseline_gsm8k`、`baseline_gpqa` 可独立运行；`gpqa_deepseek_v4_flash_accuracy` 完整执行四种策略的 GPQA case。 |
| 单机 PD | `baseline_pd_0830` suite 运行一个 Flash PD case，Prefill/Decode 各 8 卡；使用 `prefill_variable` 并覆盖输出为 256 token，4 并发。 |
| 双机 PD | Pro 默认 suite、三个 rotation suite，以及 Flash suite；各执行四种策略 × 定长/变长的 8 个 case，生成 comparison。 |

非 PD Prefill 共 6 个 case：baseline/CPP 各有定长与变长，SRF/CPP+SRF 各有变长。统一使用 Flash 模型和 `concurrency_4`；CPP、CPP+SRF case 配置手动预热，baseline、SRF 不预热。性能 `bench` 当前默认 `repeats: 1`、`temperature: 0`、`ignore_eos: true`。

| 正式数据 profile | 输入长度配置 | 每请求输出 token | 请求数 | 使用场景 |
| --- | --- | ---: | ---: | --- |
| `srf_mixed` | 1K 与 64K 各半 | 128 | 32 | 四个冒烟 case |
| `prefill_fix` | 固定 32K | 1 | 64 | 非 PD Prefill 定长 |
| `prefill_variable` | 8K–64K 高斯分布，配置均值 32K | 1 | 64 | 非 PD Prefill 变长；单机 PD case 将输出覆盖为 256 |
| `fixed_long` | 固定 64K | 2560 | 24 | 双机 PD 定长 |
| `variable_long` | 40K–80K 高斯分布，配置均值 64K | 2560 | 24 | 双机 PD 变长 |
| `functional_smoke` | 固定 1K | 128 | 8 | 保留的数据 profile，当前 case 未引用 |

这里 K 表示 1024；长度是数据生成配置，实测 token 数以 AISBench 结果为准。

## PD 部署方式

| Suite | Server profile | Prefill | Decode | 正式并发 |
| --- | --- | --- | --- | ---: |
| `baseline_pd_0830` | `pd` | 同机 0–7 卡，TP=4/PP=2 | 同机 8–15 卡，TP=8/PP=1 | 4 |
| `deepseek_v4_flash_pd_performance` | `pd_two_host` | 80.5.9.127，0–7 卡，TP=4/PP=2 | 80.5.9.128，8–15 卡，TP=8/PP=1 | 4 |
| `deepseek_v4_pro_pd_performance` 及 rotations | `pd_two_host` | 80.5.9.127，0–15 卡，TP=8/PP=2 | 80.5.17.122，0–15 卡，TP=16/PP=1 | 8 |

Flash 和 Pro 两个性能 suite 都是双机 PD；Flash 复用 Pro 命名的 case，通过 suite 覆盖为 Flash 模型、4 并发和 Flash 部署参数。单机 PD 的三个角色由框架管理；双机模式仅管理本机 Prefill/Proxy，Decode 为 external。当前 PD 显存利用率为 0.93，Decode 的 `--max-num-batched-tokens` 为 120，Decode 不显式设置 DP。

DeepSeek V4 Flash 的 127 Prefill + 128 Decode、Pro 的 127 Prefill + 122 Decode 双机部署，以及容器 `hccn.conf`、IP/网卡迁移、启动顺序、健康检查、日志监控和停止方法见 [双机 PD 性能测试教程](docs/deepseek_v4_flash_two_host_pd.md)。

## 模型、预热与扩展配置

`configs/model.yaml` 的根节点是 `models`。case 中的 `model` 引用该集合；`model_tag` 是 `vllm serve <model_tag>` 的实际路径或标识，模型专用参数放在 `arguments` 中。默认模型路径在该文件，PD 角色可在 suite 中覆盖自己的 `model_tag`。server 默认选择 `qwen3_30b_a3b_w8a8`；未显式配置 `--tokenizer` 时使用 `model_tag`。当前所有模型均显式设置最大上下文；DeepSeek 的 block size 为 64，Qwen 在模型配置中覆盖为 128。

Flash 的 PD 与非 PD 测试统一引用 `deepseek_v4_flash_a8w8_mtp`，共用模型路径、1M 上下文和多线程加载配置。Server、data、bench、accuracy、report 的 profile 名称保持各自职责，不需要随模型键改名。使用 `model_max_len` 预热的 Flash case 会按最终 `--max-model-len` 自动派生输入长度；当前默认是 1048575 token 输入加 1 token 输出。

Pro 使用 `deepseek_v4_pro_w4a8_0813`；当前两端角色路径均为 `/mnt/share/DeepSeekV4-pro-0813-w4a8`。Flash 和 Pro 的预热均为 5 个请求、并发 1；双机 PD 的全部 8 个 case（包括 baseline、SRF）均预热。PD 预热前检查 Prefill `/metrics`，要求 running/waiting 连续 3 次为零；该检查不用于非 PD 服务。

```yaml
models:
  qwen3_30b_a3b_w8a8:
    description: 用于最基本功能冒烟测试的 Qwen 模型
    model_tag: /home/weight/Qwen3-30B-A3B-W8A8
    arguments:
      --quantization: ascend
```

```yaml
profiles:
  non_pd_cpp:
    arguments:
      --additional-config:
        scheduler_config:
          profiling_chunk_config:
            enabled: true

  mtp_example:
    description: MTP 配置结构示例，参数名和值应按当前 vLLM 版本填写
    arguments:
      --mtp-argument: null
    environment:
      MTP_ENVIRONMENT_VARIABLE: "1"
```

新增 MTP 等特性时只需增加或覆盖 profile 中的 `arguments`、`environment`，无需修改 `scripts/server.py`。示例中的 MTP 参数名是占位符，不能直接作为测试配置使用。

suite 可通过 `case_overrides` 按 case 名称覆盖其配置，覆盖目标必须属于该 suite：

```yaml
baseline_smoke:
  cases: [baseline_smoke, baseline_gsm8k, baseline_gpqa]
  case_overrides:
    baseline_smoke:
      model: deepseek_v4_flash_a8w8_mtp
```

suite 不允许嵌套。相邻 case 的最终 server 配置相同时复用服务，配置不同时先关闭当前服务再启动下一服务；任一 case 或服务生命周期失败后立即停止。性能 suite 可设置 `comparison: true`，在所有 case 成功后生成按指标横向排列的 comparison JSON 和 CSV。包含性能和精度 case 的 suite 会用同一启动时间建立两个 run 目录，并在 `runs/` 根目录写一份 suite manifest。

## 运行产物

run ID 格式为：

```text
<服务器alias>_<vllm-ascend分支>_<MMDDHHMM>_<performance|accuracy>
```

`scripts/runtime_info.py` 从当前 hostname 推导服务器 alias，并从当前环境中 editable 安装的 `vllm-ascend` 仓库读取 Git 分支和 commit。每个 run 目录包含：

- `selected.yaml`：所选 case/profile 原始段与最终生效配置的单文件汇总。
- `run.json`：运行状态、命令、数据哈希和动态溯源信息。
- `server-<NN>.log`：按 suite 服务生命周期编号的 vLLM 进程组日志。
- `<case>-warmup.log`、`<case>-bench.log` 或 `<case>-accuracy.log`。
- `<case>-performance.json/csv`、`<case>-performance-details.json` 或 `<case>-accuracy.json`。
- `<case>-report.json/csv`：按 `configs/report.yaml` 指定字段严格汇总；必需字段缺失即失败。
- `<suite>-comparison.json/csv`：`comparison: true` 的性能 suite 横向指标对比。

正式数据默认持久化为 `<profile>-<config_hash>-<content_hash>.jsonl`，相邻 `.meta.json` 保存 AISBench 要求的 `request_count`、`sampling_mode` 和 `output_config`。再次使用时校验内容哈希。配置了手动预热的性能 case，会按服务 block size 检查预热与正式数据前缀不重合；精度 case 不执行这项性能数据检查。预热统计不进入正式报告。

报告汇总成功/失败请求数、吞吐、TTFT、E2EL、TPOT 或 accuracy；缺失必需指标即失败。当前未实现准确率最低分、性能回退阈值或 `Failed Requests == 0` 的额外判定，因此 `completed` 表示流程完成，不等同于指标达到验收标准。只输出 1 token 的 Prefill 测试重点看 TTFT 和 Prefill 吞吐，TPOT 是否可用取决于 AISBench 输出。

## 本地配置检查

```bash
python -m pytest -q tests/test_pd_configuration.py
```

现有测试覆盖模型 block size、公共服务 profile、Prefill 四策略 suite、PD 拓扑、预热长度与请求空闲检查。它们不启动真实 vLLM/AISBench 服务，端到端验证需在前述容器环境中执行相应 case 或 suite。
