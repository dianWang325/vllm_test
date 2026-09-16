# GLM-5.2 PP2 prefill, CPP explicitly off

- Date: 2026-09-16 (Asia/Shanghai)
- PP0/head: `80.5.17.108`; PP1/worker: `80.5.17.111`
- Container: `wd_test0825` on both hosts
- Model: `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`
- Deployment: TP16, PP2, expert parallel enabled, `max_model_len=87040`
- CPP setting: `scheduler_config.profiling_chunk_config.enabled=false` explicitly passed through `--additional-config`
- vLLM's ordinary `enable_chunked_prefill` was not changed
- Launch script: `plain_scripts/test_pp2_glm52_ep_108_111_85k_cpp_off.sh`
- Warmup: 5 requests, four with 87039 prompt tokens and one with 85937, output length 1, concurrency 1
- Fixed: 24 requests, 65536 prompt tokens, output length 1, concurrency 1
- Outcome: warmup 5/5 and fixed 24/24 succeeded. Both model processes were stopped after the benchmark; both hosts returned to 8/8 idle physical NPUs.

Fixed-stage results: 392.425653 s elapsed, 0.061158 requests/s, 4008.056018 input tokens/s, total latency p50 16.349591 s and p95 16.376460 s. Visible-token TTFT p50 16.349460 s and p95 16.380617 s, measured from 12 of 24 requests; the other 12 returned no visible text despite one completion token, so TTFT is null for them. TPOT is undefined for single-token outputs.

Warmup p50 TTFT was 20.530866 s. Its first request took 53.831157 s due to cold-start effects; the other four were 20.17–22.42 s.

The previous 110/111 run had no CPP-enabling config, and the installed vLLM Ascend default for `profiling_chunk_config.enabled` is already `false`. This run therefore verifies explicit disablement, **not** an on/off CPP performance comparison. It also uses a different PP0 host (108 instead of 110). The previous fixed-stage total latency p50 was 16.354420 s and request throughput was 0.061133 requests/s, effectively the same at this precision.

`manifest.json`, `requests.jsonl`, `warmup_summary.json`, and `summary.json` are the unmodified benchmark outputs.
