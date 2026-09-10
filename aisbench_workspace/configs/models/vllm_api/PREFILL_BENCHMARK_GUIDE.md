# AISBench Prefill 测试使用指南

本文说明如何使用以下配置向已经启动的 OpenAI 兼容模型服务发送 Prefill 性能请求：

- `prefill_warmup.py`：5 条预热请求，并发 1，输出 1 token。
- `prefill_variable.py`：24 条正式请求，并发 4，输入长度为 40K～80K 的高斯分布（均值 64K、标准差 10K），输出 2560 token。
- `prefill_fixed.py`：24 条正式请求，并发 4，输入长度固定为 64K，输出 2560 token。

warmup 输出长度为 1 token，不统计 TPOT；两个正式测试输出 2560 token，需要统计 TPOT。

## 一、适用条件

以下命令不绑定服务器 IP。在任意目标机器上进入能够执行 `docker exec` 的 Shell，然后设置实际环境参数：

```bash
CONTAINER=<工作容器名>
PROJECT=/home/w00985415/vllm_test
AIS_BENCH=/usr/local/python3.11.10/bin/ais_bench
CFG_ROOT=${PROJECT}/aisbench_workspace/configs/models/vllm_api
OUTPUT_ROOT=${PROJECT}/aisbench_workspace/outputs
```

如需从其他机器登录，服务器地址由执行者指定，例如：

```bash
ssh <用户名>@<服务器地址>
```

配置文件当前假定：

- AISBench 和模型服务位于同一个工作容器中。
- 容器内服务地址为 `http://127.0.0.1:18080`。
- 服务模型名为 `deepseek-v4-flash`。
- tokenizer 辅助目录为 `${PROJECT}/aisbench_workspace/tokenizers/deepseek-v4-flash`。

如果端口、模型名或容器内项目路径不同，需要同步修改配置文件。

## 二、配置兼容状态

当前可以直接使用：

```text
prefill_warmup.py
prefill_variable.py
```

`prefill_fixed.py` 已与另外两个配置完成相同的兼容修正，可以直接使用：

- 模型类使用 `VLLMCustomAPIStream`。
- 未传递当前 AISBench 不支持的 `stream`、`api_key` 和 `use_timestamp`。
- `DefaultPerfSummarizer` 设置当前 AISBench CLI 必需的 `attr="performance"`；CLI 会在实例化 summarizer 前消费该字段。
- `CustomDataset` 未传递不支持的 `meta_path`。
- tokenizer 的 `path` 指向 `${PROJECT}/aisbench_workspace/tokenizers/deepseek-v4-flash`，避免模型目录中的 YAML 使 AISBench 误选 `MindformersTokenizer`。

## 三、标准执行流程

标准顺序为：

```text
等待健康接口返回 200 → warmup → 确认 5/5 成功 → 正式 Prefill 测试
```

只有明确要求“不需要预热”时，才可在健康检查通过后直接执行正式测试。

### 1. 等待服务健康

```bash
while true; do
  CODE=$(docker exec "${CONTAINER}" \
    curl -sS -o /dev/null -w "%{http_code}" \
    --connect-timeout 3 --max-time 5 \
    http://127.0.0.1:18080/health 2>/dev/null || true)

  echo "health=${CODE}"
  [ "${CODE}" = "200" ] && break
  sleep 5
done
```

`000`、连接失败或其他 HTTP 状态码均不能发送测试数据。

### 2. 创建本轮独立输出目录

```bash
RUN_ROOT=${OUTPUT_ROOT}/prefill_$(date +%Y%m%d%H%M%S)
```

每轮测试必须使用新的 `RUN_ROOT`；warmup、variable 和 fixed 还需使用各自独立的 `--work-dir`。

### 3. 执行 warmup

```bash
docker exec -w "${PROJECT}" "${CONTAINER}" \
  "${AIS_BENCH}" \
  "${CFG_ROOT}/prefill_warmup.py" \
  --mode perf \
  --work-dir "${RUN_ROOT}/warmup" \
  --debug
```

继续正式测试前必须确认：

```text
Success Requests = 5
Failed Requests  = 0
```

### 4. 执行可变长度 Prefill 测试

```bash
docker exec -w "${PROJECT}" "${CONTAINER}" \
  "${AIS_BENCH}" \
  "${CFG_ROOT}/prefill_variable.py" \
  --mode perf \
  --work-dir "${RUN_ROOT}/variable" \
  --debug
```

预期结果：

```text
Success Requests = 24
Failed Requests  = 0
输入长度范围     = 40960～81920
配置分布均值     = 65536
配置标准差       = 10240
输出长度         = 2560
```

### 5. 执行固定长度 Prefill 测试

`prefill_fixed.py` 已完成兼容修正，可直接运行：

```bash
docker exec -w "${PROJECT}" "${CONTAINER}" \
  "${AIS_BENCH}" \
  "${CFG_ROOT}/prefill_fixed.py" \
  --mode perf \
  --work-dir "${RUN_ROOT}/fixed" \
  --debug
```

预期为 24 条固定 64K 输入请求、2560 token 输出，并发 4，全部成功且无失败请求。

## 四、固定运行约束

- 使用 `--mode perf --debug`。
- 不设置或依赖 `VTEST_AISBENCH_RUNTIME`。
- 不传递 `--num-warmups`。
- 无需额外传递 `--num-prompts`，请求数量由对应 JSONL 数据集决定。
- 每轮和每个阶段使用独立的 `--work-dir`。
- warmup 输出长度为 1 token，TPOT 缺失是正常现象。
- fixed/variable 正式测试输出长度为 2560 token，应检查并汇报 TPOT。

## 五、成功判定与结果检查

不能只根据 AISBench 进程退出码判断成功。当前版本可能在子任务失败时仍返回退出码 0。

必须同时满足：

1. 控制台中的 `Success Requests` 等于预期请求数。
2. `Failed Requests` 等于 0。
3. 对应的性能 JSON 和 CSV 均存在且非空。
4. CSV 中包含预期数量的逐请求记录。

查找结果文件：

```bash
find "${RUN_ROOT}" \
  -path "*/performances/vtest/vtest_data.json" \
  -o -path "*/performances/vtest/vtest_data.csv"
```

检查文件非空：

```bash
find "${RUN_ROOT}" \
  -path "*/performances/vtest/vtest_data.json" \
  -type f -size +0c

find "${RUN_ROOT}" \
  -path "*/performances/vtest/vtest_data.csv" \
  -type f -size +0c
```

结果通常位于：

```text
<work-dir>/<时间戳>/performances/vtest/vtest_data.json
<work-dir>/<时间戳>/performances/vtest/vtest_data.csv
```

部分旧版 AISBench 可能使用 `customdataset.json`/`customdataset.csv`；以实际日志打印的 `Performance Result files located in ...` 目录为准。

## 六、建议汇报指标

正式结果建议至少汇报：

- Request Throughput
- Prefill Token Throughput
- Input Token Throughput
- Output Token Throughput
- Total Token Throughput
- Success Requests / Failed Requests
- TTFT Average / Median / P90 / P95 / P99
- TPOT Average / Median / P90 / P95 / P99
- E2EL Average / Median / P90 / P95 / P99
- 总输入 Token、平均输入长度和输入长度范围
- Benchmark Duration

仅 warmup 不汇报 TPOT；fixed/variable 正式测试应汇报 TPOT。
