# DeepSeek V4 Flash/Pro 双机 PD 性能测试教程

本文对应当前工作区的双机 PD suite。`deepseek_v4_flash_pd_performance` 和 `deepseek_v4_pro_pd_performance` **都是跨节点 PD**，区别是模型、Decode 主机、卡数、并行参数和正式请求并发，而不是单机与双机。单机 PD 对应 `baseline_pd_0830`；`prefill_variable_performance` 则是非 PD Prefill 专项测试，参见 [README](../README.md)。

Proxy 和 AISBench 由 `80.5.9.127` 上的 `vtest` 管理。两台机器各自启动本机进程，框架不通过 SSH 启停远端角色。以下以 Pro 为主线，并提供 Flash 的对应命令；文档更新不表示已同步或验证远端环境。

## 1. 部署拓扑

| 项目 | Pro suite | Flash suite |
| --- | --- | --- |
| 模型 profile | `deepseek_v4_pro_w4a8_0813` | `deepseek_v4_flash_a8w8_mtp` |
| Prefill | `80.5.9.127`，卡 `0-15`，TP=8 / PP=2 / DP=1 | `80.5.9.127`，卡 `0-7`，TP=4 / PP=2 / DP=1 |
| Decode | `80.5.17.122`，卡 `0-15`，TP=16 / PP=1 | `80.5.9.128`，卡 `8-15`，TP=8 / PP=1 |
| Prefill / Decode 通信网卡 | `enp194s0f0` / `enp48s3u1u1` | 两端均为 `enp194s0f0` |
| 正式性能并发 | `concurrency_8` | `concurrency_4` |
| Case 组合 | baseline、CPP、SRF、CPP+SRF，各含 fixed / variable | 复用相同八个 case，通过 suite 覆盖模型、bench 和部署参数 |

两种配置的 Decode 均未显式设置 `--data-parallel-size`。Flash 复用的 case ID 仍带 `deepseek_v4_pro_` 前缀，不能据名称判断实际模型，应查看 suite 解析结果。双机部署请执行 suite；直接执行这些 case 不会带上 suite 中的主机、卡数等覆盖项。

Pro 的角色与端口如下（Flash 仅将 Decode 主机换为 128，并使用上表卡数）：

| 主机 | 角色 | 设备 | API 端口 | KV 端口基址 | 管理方式 |
| --- | --- | --- | --- | --- | --- |
| `80.5.9.127` | Proxy | - | `18080` | - | `vtest` 自动启动和停止 |
| `80.5.9.127` | Prefill | `0-15` | `18081` | `36000` | `vtest` 自动启动和停止 |
| `80.5.17.122` | Decode | `0-15` | `18082` | `36100` | 在 122 上用 `pd_role.py` 独立管理 |

上表使用通用的 `pd_two_host` profile；Pro 与 Flash 的主机、模型、卡数、并行和网络差异分别由对应 suite 的 `case_overrides.overrides.server` 提供。

当前模型路径和 Connector：

- Pro 模型：127 和 122 均使用 `/mnt/share/DeepSeekV4-pro-0813-w4a8`
- Flash 模型：127 和 128 均使用 `/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp`
- KV Connector：`MooncakeConnectorV1`

以下分支、提交和容器是原部署记录，供环境复现参考，并非当前 YAML 强制锁定的依赖版本；运行前需自行核对两端实际环境：

- vLLM Ascend 分支：`codex/cpp-srf-composition-pr15646`
- vLLM Ascend 已验证提交：`7e555f4384a7ddfcd4462d1a8a3cda67f556a189`
- vLLM 已验证提交：`ba07e4a48fc9`
- vllm_test 分支：`dev/pd`
- 容器：`wd_test0825`，使用 host network

`kv_connector_extra_config` 必须在 Prefill 和 Decode 两端保持完全一致，描述的是整个 PD 拓扑，而不只是本机进程。Pro 当前配置为：

```yaml
prefill:
  dp_size: 1
  tp_size: 8
  pp_size: 2
decode:
  dp_size: 1
  tp_size: 16
  pp_size: 1
```

Flash 对应 Prefill `dp_size=1 / tp_size=4 / pp_size=2`、Decode `dp_size=1 / tp_size=8 / pp_size=1`。Connector 的 `dp_size: 1` 是拓扑描述，不等于 Decode 显式传入了 DP 启动参数。

## 2. 新服务器创建容器时挂载 hccn.conf

### 2.1 为什么应在创建容器时处理

`/etc/hccn.conf` 记录本机 NPU 的通信配置，AscendDirect/Mooncake 在建立跨机 KV 传输通道时会读取它。该文件属于具体宿主机，127 和 122 的内容不应互相复制，也不建议通过 Dockerfile `COPY` 固化到通用镜像中。推荐在 `docker run` 创建容器时只读绑定本机文件，这样容器重建后配置仍然存在，宿主机配置更新后也不必重新制作镜像。

创建容器前先确认本机文件存在：

```bash
test -r /etc/hccn.conf
sha256sum /etc/hccn.conf
```

### 2.2 修改当前 docker_run.sh

当前脚本位于宿主机：

```text
/home/w00985415/proj_0825/deps/docker_run.sh
```

在现有 volume 参数中增加下面一行，注意上一行和本行末尾的反斜杠：

```diff
 -v /etc/ascend_install.info:/etc/ascend_dockerinstall.info \
+-v /etc/hccn.conf:/etc/hccn.conf:ro \
 -v /root/.cache:/root/.cache \
```

下面的 16 卡容器示例同时适用于 127 Prefill 和 122 Decode：

```bash
docker run --privileged -it -d --net=host \
  --shm-size=512g \
  --name wd_test0825 \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_dockerinstall.info \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /root/.cache:/root/.cache \
  -v /mnt:/mnt \
  -v /home:/home \
  6bad976f87e8
```

如果希望在容器内执行 HCCN 诊断，可按新服务器上工具的真实路径额外挂载，例如当前机器可用：

```bash
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool:ro \
```

这不是运行 KV Connector 的必需项，`hccn.conf` 才是本次必须持久化到容器中的文件。

设备列表应与该容器实际分配的物理卡一致。Pro 配置在 127 和 122 均使用 `davinci0` 至 `davinci15`。当前脚本含 `--privileged`，但仍建议把设备列表写准确，便于迁移和审计；最终由 `ASCEND_RT_VISIBLE_DEVICES` 决定 vLLM 角色使用的卡。

容器创建后验证挂载来自本机且为只读：

```bash
docker inspect wd_test0825 --format '{{range .Mounts}}{{if eq .Destination "/etc/hccn.conf"}}{{.Source}} -> {{.Destination}} (RW={{.RW}}){{end}}{{end}}'
docker exec wd_test0825 sha256sum /etc/hccn.conf
sha256sum /etc/hccn.conf
```

容器与宿主机的 SHA256 应一致，`RW=false`。对已经创建但没有该 mount 的容器，可临时使用 `docker cp /etc/hccn.conf wd_test0825:/etc/hccn.conf` 验证方案；这不会持久化到下次重建，因此正式使用仍应修改 `docker_run.sh`。

## 3. 在新机器上调整主机 IP 和网卡

双机公共结构和具体部署参数分别位于：

```text
configs/server.yaml
profiles.pd_two_host
configs/suites.yaml
suites.<suite 名>.case_overrides.<case 名>.overrides.server
```

迁移主机时需要检查或修改以下字段（下表为 Pro；Flash 的主机和网卡见第 1 节）：

| 位置 | 含义 | 当前值 |
| --- | --- | --- |
| `pd.prefill.endpoint_host` | Proxy 和 Decode 实际访问 Prefill 的地址 | `80.5.9.127` |
| `pd.prefill.environment.VLLM_HOST_IP` | Prefill 向 Connector 公布的本机地址 | `80.5.9.127` |
| `pd.prefill.environment.HCCL_IF_IP` | Prefill HCCL 通信地址 | `80.5.9.127` |
| `pd.decode.endpoint_host` | Proxy 和健康检查实际访问 Decode 的地址 | `80.5.17.122` |
| `pd.decode.environment.VLLM_HOST_IP` | Decode 向 Connector 公布的本机地址 | `80.5.17.122` |
| `pd.decode.environment.HCCL_IF_IP` | Decode HCCL 通信地址 | `80.5.17.122` |
| 两端 `NO_PROXY`、`no_proxy` | 防止双机流量进入 HTTP 代理 | 两台主机 IP、localhost |
| Prefill 三项通信网卡 | Gloo、TP、HCCL 使用的网卡 | `enp194s0f0` |
| Decode 三项通信网卡 | Gloo、TP、HCCL 使用的网卡 | `enp48s3u1u1` |

每台机器用下面的命令确认 IP 所属网卡：

```bash
ip -o -4 addr show
```

需要强调：

- `arguments.--host: 0.0.0.0` 是 vLLM 的监听地址，为跨主机访问而设置，通常无需随主机 IP 修改。
- `endpoint_host`、`VLLM_HOST_IP` 和 `HCCL_IF_IP` 必须填写对端可达的真实主机 IP，不能使用 `0.0.0.0`、`127.0.0.1` 或未经验证的其他网段地址。
- 两台主机应使用同一版本的完整代码和配置目录；角色启动工具会解析 suite 后提取本机角色配置。仅修改本地 Windows 工作区不会自动更新远端。
- Pro 的默认 Prefill/Tokenizer 模型路径在 `configs/model.yaml`，两个角色的主机专用路径由 Pro suite 覆盖项中的 `pd.prefill.model_tag` 和 `pd.decode.model_tag` 明确覆盖。
- Flash 模型路径由 `configs/model.yaml` 中的 `models.deepseek_v4_flash_a8w8_mtp.model_tag` 配置。
- API 与 KV 端口位于 `configs/server.yaml` 的通用 profile；卡号、DP/TP/PP、主机地址和网卡位于对应 suite 的服务覆盖项。

## 4. 网络与端口准备

本方案使用 Docker host network，不需要 Docker `-p` 映射。两台主机之间至少需要放通：

- API：`18081`（Prefill）、`18082`（Decode）；`18080` 是 127 上的 Proxy 入口。
- KV 固定端口：Pro Prefill `36000-36015`、Decode `36100-36115`。
- Mooncake Transfer Engine 动态 RPC 端口：默认 `15000-17000`。
- AscendDirect 动态端口：每个可见 NPU 使用 1000 个端口；8 卡容器通常为 `20000-27999`。若容器可见 16 张卡，保守放通 `20000-35999`。

启动前只检查端口占用，不需要扫描显存：

```bash
ss -lntp | grep -E ':(18080|18081|18082|3600[0-9]|3601[0-5]|3610[0-9]|3611[0-5])\b' || true
```

如果远端拉取代码时遇到网络问题，可以先执行：

```bash
source /home/w00985415/proxy.sh
```

## 5. 两台机器准备相同代码版本

在 Prefill 和 Decode 宿主机分别同步本次要测试的代码与配置。以下是使用已推送 `dev/pd` 分支的示例；先确认工作区无冲突，并确保所需本地修改已包含在待同步版本中：

```bash
cd /home/w00985415/vllm_test
git fetch origin
git switch dev/pd
git pull --ff-only origin dev/pd
```

容器挂载了 `/home`，因此宿主机更新后容器内会看到相同文件。再检查两个依赖仓库：

```bash
docker exec -it wd_test0825 bash -lc '
  cd /home/w00985415/proj_0825/deps/vllm-ascend &&
  git branch --show-current &&
  git rev-parse HEAD &&
  cd /home/w00985415/proj_0825/deps/vllm &&
  git rev-parse HEAD
'
```

不要在一台机器上单独修改 `kv_connector_extra_config`；Prefill、Decode 对拓扑的理解不同会导致 rank、block 或传输映射不一致。

检查模型路径；如果实际模型放在备用目录，则只修改前述 `model_tag`：

```bash
docker exec wd_test0825 test -d /mnt/weight/DeepSeek-V4-Flash-w8a8-mtp
docker exec wd_test0825 test -d /mnt/share/DeepSeekV4-pro-0813-w4a8  # 127
docker exec wd_test0825 test -d /mnt/share/DeepSeekV4-pro-0813-w4a8  # 122
```

## 6. 启动 Pro Decode（122）

先在 122 上独立启动 Decode。`pd_role.py` 会解析 suite 的最终配置并启动一个带独立进程组的本机进程；`remote_process.py` 虽沿用历史文件名，但只管理当前主机上的 PID 文件和进程组，不建立 SSH 连接。

```bash
ssh 80.5.17.122
docker exec -d \
  -w /home/w00985415/vllm_test \
  wd_test0825 \
  bash -lc 'mkdir -p runs && exec python -m scripts.pd_role run \
    --role decode \
    --suite deepseek_v4_pro_pd_performance \
    --pid-file /tmp/vtest-deepseek-v4-pro-decode.pid \
    >runs/decode-two-host.log 2>&1'
```

查看启动日志并等待健康：

```bash
docker exec wd_test0825 tail -f /home/w00985415/vllm_test/runs/decode-two-host.log
curl --fail --max-time 3 http://127.0.0.1:18082/health
```

在 127 上也应能访问该地址：

```bash
curl --fail --max-time 3 http://80.5.17.122:18082/health
```

配置中的 `pd.decode.external: true` 表示 127 的 `vtest` 只等待并使用这个 Decode，不会远程启动或停止它。

## 7. 启动 Prefill、Proxy 和性能 suite（127）

Decode 健康后，在 127 上启动完整 suite：

```bash
ssh 80.5.9.127
docker exec -it \
  -w /home/w00985415/vllm_test \
  wd_test0825 \
  ./vtest run suite deepseek_v4_pro_pd_performance
```

需要脱离终端运行时：

```bash
docker exec -d \
  -w /home/w00985415/vllm_test \
  wd_test0825 \
  bash -lc 'mkdir -p runs && exec ./vtest run suite deepseek_v4_pro_pd_performance \
    >runs/two-host-suite.launcher.log 2>&1'
```

127 上的 `vtest` 将按以下顺序工作：

1. 启动本机 Prefill。
2. 等待 Prefill `80.5.9.127:18081/health` 和外部 Decode `80.5.17.122:18082/health`。
3. 启动本机 Proxy，并通过 `127.0.0.1:18080` 接收 AISBench 请求。
4. 依次执行 baseline、CPP、SRF、CPP+SRF 的 fixed/variable 八个 case。
5. 在服务配置发生变化时管理本机服务生命周期；不会停止外部 Decode。

### 7.1 数据、预热与策略顺序

| 后缀 / 数据 profile | 正式请求数 | 配置输入长度 | 配置输出长度 |
| --- | --- | --- | --- |
| fixed / `fixed_long` | 24 | 固定 65536 | 2560 |
| variable / `variable_long` | 24 | 高斯分布，40960–81920，均值 65536，标准差 10240 | 2560 |

每个 case 均启用 `model_max_len` 预热：输入长度 1048575、输出长度 1、5 个请求、并发 1，预热统计不计入正式报告。正式请求前先检查预热与正式数据的前缀不重复，再等待 Prefill running/waiting 连续三次为零，然后执行预热和正式压测。正式并发 Pro 为 8、Flash 为 4，当前默认重复次数为 1；实际 token 数以 AISBench 输出为准。

CPP 和 SRF 仅覆盖 Prefill 调度器：CPP `smooth_factor=1.0`、`need_timing=true`；SRF `threshold=65536`、`long_max_wait_ms=2000`。同一策略的 fixed / variable 相邻执行并复用相同服务配置，切换策略时重启框架管理的服务；八个 case 对应四个服务段，external Decode 保持运行。

| Pro suite | 策略顺序（每种策略均先 fixed、后 variable） |
| --- | --- |
| `deepseek_v4_pro_pd_performance` | baseline → CPP → SRF → CPP+SRF |
| `deepseek_v4_pro_pd_performance_rotation_1` | CPP → SRF → CPP+SRF → baseline |
| `deepseek_v4_pro_pd_performance_rotation_2` | SRF → CPP+SRF → baseline → CPP |
| `deepseek_v4_pro_pd_performance_rotation_3` | CPP+SRF → baseline → CPP → SRF |

四个 Pro suite 复用同一批 case 和部署覆盖配置，只改变顺序；均生成性能对比报告。Flash 使用默认策略顺序，没有单独的 rotation suite。

运行 Pro 轮换测试时，把 Decode 和主 suite 命令中的 suite 名同时替换为本轮名称。为使各轮从独立服务状态开始，可在每个 suite 结束后使用 PID 文件停止 external Decode，再启动下一轮；框架不会自动执行该操作。

### 7.2 改为 Flash 双机测试

先结束正在占用 127 Prefill/Proxy 端口的测试。Flash Decode 在 **128** 上启动，使用独立的 PID 文件和日志：

```bash
ssh 80.5.9.128
docker exec -d -w /home/w00985415/vllm_test wd_test0825 \
  bash -lc 'mkdir -p runs && exec python -m scripts.pd_role run \
    --role decode \
    --suite deepseek_v4_flash_pd_performance \
    --pid-file /tmp/vtest-deepseek-v4-flash-decode.pid \
    >runs/decode-flash-two-host.log 2>&1'
```

确认日志正常后，在 **127** 上检查远端健康并启动 Flash suite：

```bash
curl --fail --max-time 3 http://80.5.9.128:18082/health
docker exec -it -w /home/w00985415/vllm_test wd_test0825 \
  ./vtest run suite deepseek_v4_flash_pd_performance
```

如需启动前预览实际 Decode 命令，可在仓库根目录执行（不会启动服务）：

```bash
python -m scripts.pd_role command --role decode --suite deepseek_v4_flash_pd_performance
```

## 8. 健康检查、日志和结果

三项健康检查（Flash 将 Decode 检查放在 128 上执行）：

```bash
# 127
curl --fail --max-time 3 http://127.0.0.1:18080/healthcheck
curl --fail --max-time 3 http://127.0.0.1:18081/health

# 122
curl --fail --max-time 3 http://127.0.0.1:18082/health
```

运行目录会在 127 的 `runs/` 下创建，命名格式为：

```text
<服务器alias>_<vllm-ascend分支>_<MMDDHHMM>_performance
```

常用监控命令：

```bash
# 127：定位最新运行目录、跟踪主服务日志
cd /home/w00985415/vllm_test
RUN_DIR=$(find runs -maxdepth 1 -type d -name '*_performance' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
echo "$RUN_DIR"
tail -f "$RUN_DIR/server-01.log"

# 127：检查本轮所有服务日志的关键异常
grep -Ein 'ERROR|Traceback|Exception|FATAL|Failed|Killed|OOM|Out of memory' \
  "$RUN_DIR"/server-*.log

# 查看结构化状态
python -m json.tool "$RUN_DIR/run.json"
```

在 Decode 主机单独查看日志（Pro 为 122，Flash 为 128，后者日志名为 `decode-flash-two-host.log`）：

```bash
tail -f /home/w00985415/vllm_test/runs/decode-two-host.log
grep -Ein 'ERROR|Traceback|Exception|FATAL|Failed|Killed|OOM|Out of memory' \
  /home/w00985415/vllm_test/runs/decode-two-host.log
```

报告包含各 case 性能指标及 suite 对比；Flash 的 case 文件名仍可能含 `deepseek_v4_pro_`。必需指标缺失会报错，但当前没有性能阈值或“失败请求数必须为零”的自动验收规则，应结合请求明细、指标和两端日志判断结果。

仅端口健康并不足以证明 KV 传输成功。还应在日志中看到 Pro Prefill 公布 `80.5.9.127:36000-36015`，以及 Decode 成功接收来自 Prefill 的 KV block；若出现 transfer failed、超时或 block/rank 映射错误，本轮 PD 性能数据不应直接作为有效基线。

## 9. 正常停止和失败清理

正常情况下先等待 127 上的 suite 结束。`vtest` 会停止自己启动的 Proxy 和 Prefill。随后在 122 上用精确 PID 文件停止外部 Decode：

```bash
docker exec \
  -w /home/w00985415/vllm_test \
  wd_test0825 \
  python -m scripts.pd_role stop \
    --role decode \
    --pid-file /tmp/vtest-deepseek-v4-pro-decode.pid
```

Flash 测试结束后，在 **128** 上执行对应停止命令：

```bash
docker exec -w /home/w00985415/vllm_test wd_test0825 \
  python -m scripts.pd_role stop --role decode \
    --pid-file /tmp/vtest-deepseek-v4-flash-decode.pid
```

该命令校验 PID、进程启动时间和 PGID 后再终止对应进程组，避免使用宽泛的 `pkill` 误伤其他用户服务。

如果 suite 失败：

1. 保存 127 的运行目录、`run.json` 和 `server-*.log`。
2. 保存 Decode 主机（Pro 122 / Flash 128）的独立日志。
3. 确认 127 的本机 Proxy/Prefill 已退出。
4. 使用上面的 `pd_role.py stop` 停止 Decode。
5. 修正配置后重新按 Decode → Prefill/Proxy/suite 的顺序启动。

## 10. 常见问题定位

### Prefill 启动时报 KV cache 内存不足

日志若明确提示模型最大长度需要的 KV cache 大于可用值，先确认是否存在其他显存占用，再检查 `--gpu-memory-utilization` 和 `--max-model-len`。本配置 Prefill 与 Decode 均使用 `0.93`，模型最大长度保持 `1048576`。不要仅凭 Decode 能启动就推断 Prefill 一定能启动，两端并行布局和 KV cache 需求不同。

### API 健康但 KV 传输失败

依次核对：

1. 两端 `kv_connector_extra_config` 是否逐字段相同。
2. `VLLM_HOST_IP` 是否为对端可达的正确主机 IP。
3. `/etc/hccn.conf` 是否来自各自宿主机且容器内可读。
4. 网卡名与 `HCCL_IF_IP` 是否匹配。
5. 固定端口和动态端口范围是否放通或被占用。
6. 两端 vLLM、vllm-ascend 和 vllm_test 是否处于同一预期版本。

KV 传输失败后不能假设 Decode 会无损回退为普通非 PD 推理；即使请求最终有输出，TTFT、吞吐和计算路径也可能已经偏离测试目标，因此应把出现传输失败的性能结果视为受影响并重新测试。

### 修改了 IP 但仍连接旧地址

检查 `configs/suites.yaml` 中对应 suite 服务覆盖项的所有 `endpoint_host`、`VLLM_HOST_IP`、`HCCL_IF_IP`、`NO_PROXY/no_proxy`，并确认两台机器已经拉取同一提交。`configs/server.yaml` 中的 `--host` 保持 `0.0.0.0`，它不是对端连接目标。

## 11. 新服务器迁移检查表

- [ ] 两台主机分别准备自己的 `/etc/hccn.conf`。
- [ ] `docker_run.sh` 增加 `/etc/hccn.conf:/etc/hccn.conf:ro`。
- [ ] 容器设备列表与分配卡号一致。
- [ ] 两台容器中的 vLLM、vllm-ascend、vllm_test 版本一致。
- [ ] `configs/suites.yaml` 中 Prefill/Decode IP、网卡和卡号已更新，`configs/server.yaml` 中公共端口正确。
- [ ] `configs/model.yaml` 及 suite 角色覆盖后的实际模型路径在各自主机存在。
- [ ] 两端 `kv_connector_extra_config` 完全一致。
- [ ] API、KV 固定端口和动态端口范围可达。
- [ ] 先在 Decode 主机本地启动 Decode，再在 Prefill 主机启动 suite。
- [ ] 三个健康接口均成功，且日志确认实际 KV block 传输成功。
- [ ] suite 完成后保存 `run.json`/报告并精确停止外部 Decode。
