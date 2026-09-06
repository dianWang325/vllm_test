# vllm_test

这是一个面向 vLLM Ascend 的最小测试框架：负责启动 vLLM 服务，使用 AISBench 构造正式数据并发送性能或精度请求，最后从 AISBench 结构化结果中生成报告。普通服务由框架完整管理；PD 配置也可以把某个角色标记为 `external: true`，由另一台主机使用同一份配置独立启动。框架不自动重试失败请求，也不提供旧配置兼容或回退路径。

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
└── tests/README.md             # 后续测试占位
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
```

编辑 `configs/*.yaml` 即可增加 model、profile、case 或 suite。每项均可填写 `description`；该字段只用于说明和 `selected.yaml`，不会参与服务、数据或请求参数。服务配置按 `server defaults < server profile < model < case overrides < suite case_overrides` 递归合并，列表和标量直接替换。

`configs/server.yaml` 中的 `arguments` 和 `environment` 会直接传给 vLLM 服务。`arguments` 的键是完整参数名：值为 `null` 或 `true` 时只输出参数名，`false` 时不输出该参数，字符串或数字作为下一个命令行参数，字典或列表编码为紧凑 JSON。`environment` 的值必须是字符串或数字；`unset_environment` 用于明确删除继承的环境变量。普通服务使用 `--host` 和 `--port` 进行健康检查和请求；跨主机 PD 角色可额外设置 `endpoint_host` 作为其他节点实际访问的地址，并用 `external: true` 表示该角色仅做健康检查、不由当前 vtest 进程启动或停止。基础 profile 不传递 `--additional-config`，CPP 和 SRF 只在各自 profile 中声明所需的调度配置。

## 双机 PD 性能测试

DeepSeek V4 Flash 的 130 Prefill + 129 Decode 双机部署、容器 `hccn.conf` 挂载、IP/网卡迁移、启动顺序、健康检查、日志监控和停止方法见 [双机 PD 性能测试教程](docs/deepseek_v4_flash_two_host_pd.md)。

`configs/model.yaml` 的根节点是 `models`。case 中的 `model` 是该集合中的模型名称；模型项中的 `model_tag` 是 `vllm serve <model_tag>` 使用的实际路径或模型标识，模型专用的 vLLM 参数仍放在 `arguments` 中。模型路径只保存在该文件。server 默认选择 `qwen3_30b_a3b_w8a8`，case 省略 `model` 时继承它；未显式配置 `--tokenizer` 时，框架与 vLLM 一样使用 `model_tag`。server 默认配置不传递 `--max-model-len`，由 vLLM 从模型自身配置读取；需要限制某个模型时，在该模型的 `arguments` 中显式设置。

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
  cpp:
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
cpp_validation:
  cases: [cpp_smoke, cpp_fixed_long]
  case_overrides:
    cpp_smoke:
      model: deepseek_v4_flash
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

正式数据默认持久化为 `<profile>-<config_hash>-<content_hash>.jsonl`，相邻 `.meta.json` 只保存 AISBench 要求的 `request_count`、`sampling_mode` 和 `output_config`。CPP 与 CPP+SRF case 会先发送独立手动预热请求，预热统计不进入正式报告；发送前会按服务 block size 检查预热与正式数据前缀不重合。
