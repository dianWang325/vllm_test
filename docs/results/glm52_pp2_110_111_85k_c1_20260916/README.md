# GLM-5.2 cross-node PP2 prefill, fixed concurrency 1

- Date: 2026-09-16 (Asia/Shanghai)
- PP0/head: `80.5.17.110`; PP1/worker: `80.5.17.111`
- Container: `wd_test0825` on both hosts
- Model: `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`
- Deployment: TP16, PP2, expert parallel enabled, `max_model_len=87040`
- Launch script: `plain_scripts/test_pp2_glm52_ep_110_111_85k.sh`
- Warmup: 5 requests, four with 87039 prompt tokens and one with 85937, output length 1, concurrency 1
- Fixed: 24 requests, 65536 prompt tokens, output length 1, concurrency 1
- Outcome: warmup 5/5 and fixed 24/24 succeeded. Both model processes were stopped after the benchmark; both hosts returned to 8/8 idle physical NPUs.

Fixed-stage results: 392.586496 s elapsed, 0.061133 requests/s, 4006.413918 input tokens/s, total latency p50 16.354420 s and p95 16.396551 s. Visible-token TTFT p50 16.358329 s and p95 16.389606 s, measured from 12 of 24 requests; the other 12 returned no visible text despite one completion token, so TTFT is null for them. TPOT is undefined for single-token outputs.

Warmup p50 TTFT was 20.511265 s. Its first request took 85.842712 s due to cold-start effects; the other four were 20.16–22.13 s.

`manifest.json`, `requests.jsonl`, `warmup_summary.json`, and `summary.json` are the unmodified benchmark outputs.
