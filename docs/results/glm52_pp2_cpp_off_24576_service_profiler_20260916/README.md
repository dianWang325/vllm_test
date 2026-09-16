# GLM-5.2 TP16/PP2/EP, CPP off, 24576: MS Service Profiler attempt

Date: 2026-09-16. Git branch: `test/glm`; launcher revision: `eed576d`.

## Configuration

- Model: `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`; TP=16, PP=2, EP enabled, `max-model-len=87040`.
- `max-num-batched-tokens=24576`; CPP explicitly disabled via `scheduler_config.profiling_chunk_config.enabled=false`.
- Both PP0 and PP1 loaded `plain_scripts/profiling/service_profiling_symbols.yaml` at startup. Their separate JSON runtime copies stayed at `enable: 0` throughout, as required for pre-warmup startup. `VLLM_TORCH_PROFILER_DIR` was unset.
- Launcher: `plain_scripts/test_pp2_glm52_ep_110_111_85k_cpp_off_24576_prof.sh`, with `PP0_HOST`, `PP1_HOST`, `NET_IFACE`, and `RUN_ID` overrides.

## Attempts and result

| PP0 / PP1 | Result |
| --- | --- |
| 80.5.17.110 / 80.5.17.111 | Initialization failed: PP0 free HBM 50.25/50.62 GiB, below the 0.85 target (~52.08 GiB). A separate `VLLM::Worker_DP` service occupied the cards. |
| 80.5.17.111 / 80.5.17.122 | Model loaded, but PP1 startup warmup timed out in an HCCL AIV `ALLTOALLV` task (rank size 16, vector core timeout after ~204 s). Machine 122 was excluded from all subsequent attempts at the user's request. |
| 80.5.17.111 / 80.5.17.120 | Initialization failed: another `VLLM::Worker_DP` service occupied ~24 GiB per chip; PP1 free HBM fell below the 0.85 target. That service was not stopped. |
| 80.5.17.111 / 80.5.9.129 | API reached HTTP 200 and both stages completed startup. The first 87,039-token prefill warmup request triggered `hcclCommInitRootInfoConfig` error code 19 on PP1; HCCL reported `Communication_Error_Initialize_Transport(EI0009): Device 0 transport init error. Reason: The network port is down.` Subsequent requests returned HTTP 500. Warmup success: 0/5. |

The 111/129 host IPs could ping each other, but that did not establish NPU/HCCL transport availability. No `prefill_fix` requests were sent. Because warmup did not succeed, neither profiler runtime JSON was switched to `enable: 1`; no MS Service Profiler trace or request latency/TTFT measurement was produced. The two JSON source files in Git remain `enable: 0`.

## Preserved evidence

- 110/111: `/home/w00985415/pp2_glm_ep_110_111_85k_cpp_off_24576_prof_20260916/` on each corresponding host; the first failed launch is also saved as `head_attempt1.log` / `worker_attempt1.log`.
- 111/122: `/home/w00985415/pp2_glm_ep_111_122_85k_cpp_off_24576_prof_20260916/` on the corresponding hosts.
- 111/120: `/home/w00985415/pp2_glm_ep_111_120_85k_cpp_off_24576_prof_20260916/` on the corresponding hosts.
- 111/129: `/home/w00985415/pp2_glm_ep_111_129_85k_cpp_off_24576_prof_20260916/` on the corresponding hosts. The PP0 `perf_warmup/summary.json` and `requests.jsonl` record all five failed requests.

All inference processes launched for this attempt were stopped or had exited; 111 and 129 were checked to have no NPU processes afterward. The `wd_test0825` containers started on 120 and 122 were not stopped; neither runs an inference service from this attempt.

This run does not establish that CPP-off itself caused the failure: the host pair and failure mode changed across attempts. A valid CPP-on/off performance comparison and dual-node profiling require a pair with available cards and a working NPU transport path.
