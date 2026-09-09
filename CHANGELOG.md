# 提交修改记录

从本次提交开始，每次提交都在本文件顶部新增一条记录，随对应代码一起提交。记录应包含日期、唯一记录 ID、修改内容及验证结果。

Git 提交正文使用 `Change-Record: <记录 ID>` 关联本文件条目。可通过下面的命令查询对应提交的实际 SHA、作者及修改文件：

```bash
git log --all --fixed-strings --grep="Change-Record: 2026-09-08-01" --format=fuller --stat
```

## 2026-09-09｜2026-09-09-01

提交主题：`feat(pd): add MTP and DSpark multi-host suites`

修改内容：

- 新增 Flash MTP 与 Pro DSpark 四机 PD suite，复用现有 P 双机 TP16/PP2、D 双机 DP2/TP16 拓扑。
- MTP 使用 `num_speculative_tokens: 1`，DSpark 使用 `num_speculative_tokens: 5`，配置同时下发 P0/P1/D0/D1。
- 保持 `MooncakeConnectorV1`、KV 端口、engine ID 和节点编排不变。

验证：

- `python -B -m pytest -q -p no:cacheprovider tests`：35 passed。
- 核对两个 suite 的最终解析结果，确认四个节点均获得对应 speculative config，Connector 均为 `MooncakeConnectorV1`。

## 2026-09-08｜2026-09-08-05

修改主题：`feat(pd): manage multi-host prefill and data-parallel decode`

修改内容：

- 在 `dev/multi-pd` 分支统一 PD 的 role/nodes 配置展开，角色参数在 case/suite 覆盖后下发所有物理节点；`pd_role` 支持同时选择 suite、case、node。
- 复用现有远端进程脚本，通过 SSH 与已有 Docker 容器管理节点；四节点先发起启动，再检查 API 健康。P1 headless 不进入 Proxy，D0/D1 均注册，Proxy 检查完整 1P/2D 数量。
- 统一预热、性能和精度期间的受管进程检查；节点失败终止当前测试，按节点记录日志和退出码，策略段结束统一停止全部 rank。
- 新增 `deepseek_v4_pro_multi_pd_performance_4case`：P DP1/TP16/PP2、D DP2/TP16/PP1，各机 16 卡；保留原有 suite。新部署 IP、SSH 目标和网卡使用显式占位值。
- `MooncakeConnectorV1`、KV 端口、engine ID 不变；新 suite 仅填写新并行布局的 KV 拓扑元数据。不增加版本探测、Connector 替换、自动重试、部署回退或全局清理。
- 逐模块检视并合并重复环境命令与存活检查，删除重复清理、重复启动命令记录及后台读取线程。社区依据和使用步骤见 `docs/multi_pd.md`。

验证：

- `python -B -m pytest -q -p no:cacheprovider tests`：35 passed，覆盖配置继承、原有 PD 回归、多端点 Proxy、启动/停止顺序、控制端 EOF、节点故障与 suite 生命周期。
- 手动生成 P1 CPP+SRF、D1 baseline 命令，确认社区跨机参数、P1 headless、D1 DP rank 及 `MooncakeConnectorV1` 均正确。
- 未连接远端、未启动模型或运行真实 AISBench；四机互联、NPU 显存及 OOM 改善效果待配置实际机器后实测。

## 2026-09-08｜2026-09-08-04

提交主题：`fix(pd): lower prefill batch tokens to 8k and move decode to 127`

修改内容：

- 将 `configs/server.yaml` 中两组 PD 配置的 Prefill `--max-num-batched-tokens` 从 `16384` 进一步降至 `8192`（8K），以降低 profile 激活峰值并争取更多 KV cache 预算；4case 及其他引用这些共享配置的 suite 均会生效。
- 将 `deepseek_v4_pro_pd_performance_4case` 的 Decode IP 从 `80.5.9.143` 改为 `80.5.9.127`，同步更新 `endpoint_host`、`VLLM_HOST_IP`、`HCCL_IF_IP` 及两端的 `NO_PROXY`、`no_proxy`；Prefill 保持 `80.5.9.139`。

验证：

- 核对本地配置差异；未运行模型启动或性能测试，显存预算及服务启动结果待实测。

## 2026-09-08｜2026-09-08-03

提交主题：`fix(pd): tune memory budgets and update four-case decode host`

修改内容：

- 将 `configs/server.yaml` 中两组 PD 配置的 `--gpu-memory-utilization` 从 `0.93` 降至 `0.90`，减少显存预算。
- 将 Prefill 节点的 `--max-num-batched-tokens` 从 `20480` 降至 `16384`，以缓解启动 profile 阶段的激活显存压力；实际 OOM 改善效果待重新运行验证。
- 同步记录本次工作区已有调整：Decode 节点的 `--max-num-batched-tokens` 从 `120` 增至 `320`。
- 将 `deepseek_v4_pro_pd_performance_4case` 的 Decode IP 从 `80.5.9.129` 改为 `80.5.9.143`，同步更新 `endpoint_host`、`VLLM_HOST_IP`、`HCCL_IF_IP` 及两端的代理排除列表，并移除旧 IP 网卡注释；Prefill 保持 `80.5.9.139`。

验证：

- 核对本次配置差异；未运行模型启动或性能测试，不将参数调整视为 OOM 已解决。

## 2026-09-08｜2026-09-08-02

提交主题：`fix(pd): move four-case suite to 108 and 109`

修改内容：

- 将 `deepseek_v4_pro_pd_performance_4case` 的 Prefill 从 `80.5.9.133` 改为 `80.5.17.108`，Decode 从 `80.5.9.138` 改为 `80.5.17.109`。
- 同步修改两端的 `endpoint_host`、`VLLM_HOST_IP`、`HCCL_IF_IP`、`NO_PROXY` 和 `no_proxy`，两端通信网卡 `GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` 统一改为 `enp48s3u1u1`。
- 更新 suite 配置内的概览及网卡注释；四个 case 的顺序、单轮执行、模型、预热及并行布局保持不变。
- 按本次要求保留 README 内容；其 4case 主机信息仍为上一版，当前部署参数以 `configs/suites.yaml` 为准。

验证：

- 本地配置测试通过：`python -B -m pytest -q -p no:cacheprovider tests/test_pd_configuration.py`，结果为 `20 passed`。
- 四个 case 的最终配置及角色启动参数检查通过，确认新 IP、网卡与代理白名单全部生效；其他 suite 和 README 无变化。
- 本次范围为本地修改与提交。

## 2026-09-08｜2026-09-08-01

提交主题：`feat(pd): add four-case Pro suite for 133 and 138`

修改内容：

- 降低 Pro 最大上下文长度至约 85K：`deepseek_v4_pro_w4a8_0813` 的 `--max-model-len` 从 `1048576` 改为 `87295`，沿用本次工作区实际配置值。所有引用此模型的 Pro case 生效；`model_max_len` 预热相应变为 `87294` token 输入、`1` token 输出。
- 新增 `deepseek_v4_pro_pd_performance_4case`，描述为“DeepSeek V4 Pro W4A8 0813 的双机 PD baseline与 CPP+SRF 性能对比【简易版】”。按以下顺序运行四个 case，各正式测试一轮（`repeats: 1`），保留独立预热和 comparison 报告：

  1. `deepseek_v4_pro_pd_baseline_fixed`
  2. `deepseek_v4_pro_pd_baseline_variable`
  3. `deepseek_v4_pro_pd_cpp_srf_fixed`
  4. `deepseek_v4_pro_pd_cpp_srf_variable`

- 新 suite 使用 Prefill `80.5.9.133` 和 external Decode `80.5.9.138`，同步设置 `endpoint_host`、`VLLM_HOST_IP`、`HCCL_IF_IP`、`NO_PROXY` 和 `no_proxy`。
- 根据两台主机实际 IP 所属网卡，将新 suite 两端的 `GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` 均配置为 `enp194s0f0`。
- 保留 Pro 原并行布局（Prefill TP=8/PP=2、Decode TP=16/PP=1）、每端 16 卡及正式并发 8；四个 case 共两个服务段。
- 更新 README 中的 suite 概览、部署参数和启停命令，修正部署教程及配置测试中的 Pro 上下文与预热长度。

验证：

- 本地配置测试通过：`python -B -m pytest -q -p no:cacheprovider tests/test_pd_configuration.py`，结果为 `20 passed`。
- 新 suite 配置解析及角色启动参数检查通过：四个 case、两段服务、单轮测试、Pro 长度、预热长度、主机 IP 和网卡配置。
- 双机代码同步不包含启动性能测试或重启模型服务。
