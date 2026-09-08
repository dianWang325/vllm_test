# 提交修改记录

从本次提交开始，每次提交都在本文件顶部新增一条记录，随对应代码一起提交。记录应包含日期、唯一记录 ID、修改内容及验证结果。

Git 提交正文使用 `Change-Record: <记录 ID>` 关联本文件条目。可通过下面的命令查询对应提交的实际 SHA、作者及修改文件：

```bash
git log --all --fixed-strings --grep="Change-Record: 2026-09-08-01" --format=fuller --stat
```

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
