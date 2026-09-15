# 两台 P 节点 PD 对比实验：启动受阻

- 代码：三台 `wd_test0825` 容器的 `vllm_test` 均同步至远端 `back136` 提交 `35db35aa1a8e436d588528b3e20173b8602ab179`。
- 主机：P0 `80.5.9.127`，P1 `80.5.9.128`，D `80.5.9.129`；每台在启动前均有 16 张空闲 NPU、指定容器和模型路径。
- 目标：P 跨两机 DP2/TP8/PP2，`enable_dsa_cp=true`；D 单机 DP2/TP8/PP1。八轮顺序见 `manifest.json`，增强组须先执行 `prefill_warmup.py`。
- 结果：**0/8 轮完成**。没有发送预热或正式 AISBench 请求，不能给出 TTFT、TPOT、tokens/s 或组间对比。

最终尝试中，两台 P 机均有 16 个 worker 分别占用 16 张卡，日志包含 DP0/DP1 的 PP0 与 PP1。权重加载完成后，P1 在 `determine_available_memory → profile_run → model.forward` 阶段报错：

```
vllm_ascend/models/deepseek_v4/model.py:940
input_ids = sp_shard(input_ids)  # TODO: support PP with dsacp.
AttributeError: 'NoneType' object has no attribute 'shape'
```

原因是 PP1 收到的 `input_ids` 为 `None`，而 DSA-CP 的序列并行路径仍尝试对它分片。仅跳过该行不能保证正确：模型的 hash MoE 路由可能仍需 `input_ids`，PP 传输和分片语义也需要完整适配。因此未生成可能错误的性能结论。

`attempt1_bad_mapping`、`attempt2_external_dp_nnodes2`、`attempt3_logical_nodes4`、`attempt4_p1_missing_dp` 和 `01_enhanced_variable` 保存了各次启动日志。为分离 DP 设备，曾在 P0/P1 临时使用 `patched_dp_device_ids.py`；原文件保存在本目录，结束后两机均已恢复到原 SHA256 `ae50fd98d763eb738cbf831b52f4b8a51db7cdc17ecd6a9e2e73fa7a4e4e0be5`。三台机器的 NPU 已重新空闲。

继续完成原定实验需要一版经过验证、同时支持 DeepSeek V4 的 PP2 与 DSA-CP 的 vllm-ascend 实现。取得该实现后，应先完成一次服务健康检查和小规模正确性验证，再从第 1 轮开始执行八轮评测。其他方案需更改用户确定的 P 配置：关闭 `enable_dsa_cp`，或取消 PP2。
