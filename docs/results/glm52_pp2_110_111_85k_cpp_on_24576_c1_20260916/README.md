# GLM-5.2 PP2 prefill, CPP on, 24576 batched tokens

- Date: 2026-09-16 (Asia/Shanghai)
- PP0/head: `80.5.17.110`; PP1/worker: `80.5.17.111`
- Container: `wd_test0825` on both hosts
- Model: `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`
- Deployment: TP16, PP2, expert parallel enabled, `max_model_len=87040`
- CPP setting: `scheduler_config.profiling_chunk_config.enabled=true` through `--additional-config`
- `max_num_batched_tokens=24576`; `weight_nz_mode=2` retained
- Launch script: `plain_scripts/test_pp2_glm52_ep_110_111_85k_cpp_on_24576.sh`
- Warmup: 5 requests, four with 87039 prompt tokens and one with 85937, output length 1, concurrency 1
- Fixed: 24 requests, 65536 prompt tokens, output length 1, concurrency 1
- Outcome: warmup 5/5 and fixed 24/24 succeeded. Both model processes were stopped after the benchmark; both hosts returned to 8/8 idle physical NPUs.

CPP selected `ProfilingChunkScheduler` with `base_chunk=24576` and completed 64 startup profiling samples in about 205 s before API readiness.

Fixed-stage results: 419.888831 s elapsed, 0.057158 requests/s, 3745.905783 input tokens/s, total latency p50 17.491287 s and p95 17.554790 s. Visible-token TTFT p50 17.491162 s and p95 17.554665 s, measured from all 24 requests. TPOT is undefined for single-token outputs.

Warmup p50 TTFT was 21.575206 s. The five warmup TTFTs ranged from 21.013388 to 22.683047 s.

The earlier 110/111 run with CPP off and `max_num_batched_tokens=16384` had fixed-stage total latency p50 16.354420 s and throughput 0.061133 requests/s. This run changed **both** CPP and the batch-token limit, so the observed difference cannot be attributed to CPP alone.

`manifest.json`, `requests.jsonl`, `warmup_summary.json`, and `summary.json` are the unmodified benchmark outputs.
