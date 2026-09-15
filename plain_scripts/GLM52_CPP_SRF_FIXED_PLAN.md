# GLM-5.2 CPP+SRF fixed-input experiment

## Service topology

- Model: `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`
- Prefill: DP1/PP2/TP16 on two hosts, 16 NPUs per host
- Decode: DP1/PP1/TP16 on one host, 16 NPUs
- `max-model-len`: 87040 on P and D
- DSA-CP: disabled on P
- Speculative decoding: disabled on P and D
- P scheduler: profiling chunk config (CPP) and short request first (SRF) enabled
- D scheduler and compilation settings: inherited from the current D service except for the requested model and topology changes

## Test scope

1. Select three compatible idle hosts: P0, P1, and D. Verify 16 free NPUs, the running or startable `wd_test0825` container, and the GLM model path on every host.
2. Replace each disposable remote `vllm_test` worktree with the latest `origin/back136`, then copy these experiment scripts/configs to it.
3. Start P0, P1, and D once; start the existing PD request path/proxy if required by the environment; wait until the OpenAI-compatible endpoint is ready.
4. Run only `aisbench_workspace/configs/models/vllm_api/prefill_fixed_glm52.py` once. Do not run warmup or the variable-input case in this experiment.
5. Preserve service, proxy, and AISBench logs plus the TTFT, TPOT, and token-throughput result files, then stop all experiment processes.

The fixed-input config keeps `max_out_len=2560`, `batch_size=4`, unlimited request rate, and `ignore_eos=True`.
