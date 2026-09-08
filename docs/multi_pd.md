# P/D 各双机部署

入口：`deepseek_v4_pro_multi_pd_performance_4case`，共四台机器，每台 16 张 NPU。复用现有 baseline / CPP+SRF 定长、变长四个 case；相邻定长/变长共用服务，切换策略时停止并重启四个节点和 Proxy。

| 机器 | 角色布局 | 本机 rank | HTTP 服务 |
| --- | --- | --- | --- |
| P0 | DP1 / TP16 / PP2，跨机 MP | `--node-rank 0` | 18081 |
| P1 | 与 P0 同一个 Prefill 实例 | `--node-rank 1 --headless` | 无 |
| D0 | DP2 / TP16 / PP1，external DP | `--data-parallel-rank 0` | 18082 |
| D1 | 与 D0 同一 DP 组，启用 EP | `--data-parallel-rank 1` | 18082 |

Proxy 在运行 `vtest` 的控制端监听 `127.0.0.1:18080`，注册 P0、D0、D1；P1 只参与计算，不注册 Proxy，不请求 `/health` 或 `/metrics`。控制端可以位于其中一台机器的测试容器内。

## 主社区依据与保留项

- P 使用主社区 [Pipeline Parallelism 的双机 MP 部署](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/pipeline_parallel.html#two-node-deployment-with-multiprocessing)：两台都设置 `--nnodes 2`、同一 master 地址和端口、同一 TP/PP，只有 rank 1 添加 `--headless`。本项目按每机 16 卡采用 TP16/PP2。
- D 的启动参数参考主社区 [DeepSeek V4 Pro 四机 external-DP 测试配置](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/multi_node/external_dp/config/DeepSeek-V4-Pro-w4a8-1M-PD.yaml)：使用 `--data-parallel-size 2`、`--tensor-parallel-size 16`、共享 DP 地址和 RPC 端口，各机器独立指定 DP rank。D 不设置 P 使用的 `--nnodes`、`--node-rank`、`--headless`。
- 网络及 Proxy 多端点参考主社区 [Mooncake Multi Node](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)。需要提前准备 NPU 互联、通信网卡及容器中的 `hccn.conf`；不要把管理口 IP 当作 NPU 互联已经配置成功。

上述资料用于确定跨机参数及网络要求，不整套移植社区模型示例：**Connector 固定保留已验证的 `MooncakeConnectorV1`**，P/D 的 `kv_port` 保留 36000/36100，`engine_id` 保留 0/1。只将 `kv_connector_extra_config` 的拓扑填写为 `prefill={dp_size:1,tp_size:16,pp_size:2}`、`decode={dp_size:2,tp_size:16,pp_size:1}`，四节点一致。不增加 Connector 自动选择、版本探测、Ray 回退或自动重试。

继续使用当前 Pro 模型参数、87295 最大上下文、87294 token 预热输入、8 并发、正式单轮、P/D batch tokens 8192/320、显存利用率 0.90；不启用新的推测解码或自动层切分调优。DP2 配合 EP 扩展专家分片，不能据此宣称所有显存占用减半；OOM 是否消失需 NPU 实测。

## 配置与环境准备

在 `configs/suites.yaml` 的新 suite 中填写以下值。`P0_IP` 等是普通占位字符串，不会自动读取环境变量。

- 四台机器的 `P0_IP/P1_IP/D0_IP/D1_IP`，同时替换 `multi_pd_no_proxy` anchor 中对应字符串；该值同时用于四节点和 Proxy 的大小写环境变量。YAML anchor 复用 endpoint、master/DP 地址及 `VLLM_HOST_IP/HCCL_IF_IP`。
- 每台机器实际承载通信 IP 的 `*_IFNAME`，用于 `GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME/HCCL_SOCKET_IFNAME`。
- `remote.host` 的四个 SSH 目标 `*_SSH`，可以是 `user@IP` 或控制端 SSH config 中的别名。控制端需要可用的 SSH 命令、非交互免密登录和已确认的 host key；远端用户须能执行 Docker。
- `remote.container` 和 `remote.workdir`，默认示例为已有的 `wd_test0825` 容器及 `/home/w00985415/vllm_test`。框架仅通过 `ssh → docker exec -i → python` 运行进程，不创建容器或同步代码。容器中的 Python 环境需已能直接运行 vLLM。
- 所有容器预先同步本分支的 `scripts/remote_process.py`，并具备一致的 vLLM/vllm-ascend 环境与权重。控制端还需要框架现有的 vllm-ascend 运行信息依赖、AISBench、数据工具、tokenizer/模型路径及配置指定的社区 Proxy 脚本。

各机 `ASCEND_RT_VISIBLE_DEVICES` 都是本机 `0–15`。P master 使用 P0:29500，D DP RPC 使用 D0:29510；同时放通实际服务、HCCL 和 Mooncake 所需端口。KV 端口配置的是基准端口，不能仅凭基准值推断通信只使用一个端口。

配置合并顺序为 server defaults → profile → model → case → suite → role → node。role 保存两台共用的并行参数和调度配置，`nodes` 只保存该机器的地址、rank、网卡及远程执行位置。基础 PD profile 也显式使用 `nodes: {p0: {}}` / `nodes: {d0: {}}`，没有另建单机兼容路径。

## 启动与检查

先在控制端查看指定节点最终生成的命令；`command` 不启动服务，也不会连接 SSH：

```bash
python -m scripts.pd_role command --role prefill --node p1 \
  --suite deepseek_v4_pro_multi_pd_performance_4case \
  --case deepseek_v4_pro_pd_cpp_srf_fixed

python -m scripts.pd_role command --role decode --node d1 \
  --suite deepseek_v4_pro_multi_pd_performance_4case \
  --case deepseek_v4_pro_pd_baseline_fixed
```

使用 `--suite` 与 `--case` 可同时保留部署覆盖及指定 case 的 CPP/SRF 参数。只给 `--case` 不会加载 suite 的四机部署；多节点角色必须指定 `--node`。`pd_role run` 是在当前容器本地运行所选节点，不执行远程派发。

完整 suite 只需在控制端执行一次，不要提前手动启动四个后端。若控制端设置了 HTTP 代理，也需将实际四机 IP 加入控制端 `NO_PROXY/no_proxy`，让 Python 健康检查直连服务（服务配置的 environment 只传给子进程）：

```bash
./vtest run suite deepseek_v4_pro_multi_pd_performance_4case
```

执行链路：展开四节点最终配置 → 发起所有 SSH 启动 → 等待 P0/D0/D1 健康且全部受管进程存活 → 启动 Proxy 并确认 1P/2D → 生成数据、检查 P0 空闲、预热、正式请求、报告 → 停止全部节点 → 下一策略段。

预热、性能及精度请求共用受管节点存活检查；P1 或任一 D 进程退出会停止当前请求进程组并终止 suite。`external: true` 的原有外部节点仍不归控制端管理；新 suite 的四节点都受管，D 已显式覆盖为 `external: false`。

## 停止与产物

正常结束及异常退出都由 suite 控制层统一停止。远端进程包装器监听 SSH/Docker 转发的标准输入 EOF，控制端关闭通道后向对应 vLLM 进程组发送 SIGTERM；即使停止发生在启动阶段，也不依赖 PID 文件恰好已落盘。四个 rank 先全部收到停止通知，再分别等待退出。超时会报错，不强杀、不做全局 `pkill`。

节点日志保存在控制端运行目录：`server-01-prefill.p0.log`、`server-01-prefill.p1.log`、`server-01-decode.d0.log`、`server-01-decode.d1.log` 和 `server-01-proxy.log`。`server-01.log` 记录启动命令，`run.json` 的 `servers[].nodes` 记录日志和进程返回码，`prefill_quiescence` 按 P API 节点名保存。失败启动也保留已创建的节点日志索引。

每次启动使用唯一 PID 文件，实际路径在启动命令的 `--pid-file` 中。网络故障造成控制端无法确认退出时，恢复连接后，在对应容器执行已有的精确停止命令：

```bash
python -m scripts.remote_process stop --pid-file /tmp/vtest-实际会话-节点序号.pid
```

本地验证运行 `python -B -m pytest -q -p no:cacheprovider tests`；测试使用模拟进程和 HTTP 响应，不加载模型，不代表四机 NPU 联调或 OOM 验收已经完成。
