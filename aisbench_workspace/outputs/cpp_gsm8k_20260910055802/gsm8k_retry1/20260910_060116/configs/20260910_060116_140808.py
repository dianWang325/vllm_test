DATASET_PATH = '/home/w00985415/vllm_test/aisbench_workspace/datasets/gsm8k'
TOKENIZER_PATH = '/home/w00985415/vllm_test/aisbench_workspace/tokenizers/deepseek-v4-flash'
datasets = [
    dict(
        abbr='gsm8k',
        eval_cfg=
        'ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt.gsm8k_eval_cfg',
        infer_cfg=
        'ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt.gsm8k_infer_cfg',
        path='/home/w00985415/vllm_test/aisbench_workspace/datasets/gsm8k',
        reader_cfg=
        'ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt.gsm8k_reader_cfg',
        type='ais_bench.benchmark.datasets.GSM8KDataset'),
]
eval = dict(
    partitioner=dict(
        out_dir=
        '/home/w00985415/vllm_test/aisbench_workspace/outputs/cpp_gsm8k_20260910055802/gsm8k_retry1/20260910_060116/results/',
        type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        debug=True,
        max_num_workers=1,
        max_workers_per_gpu=1,
        task=dict(
            dump_details=True,
            type='ais_bench.benchmark.tasks.openicl_eval.OpenICLEvalTask'),
        type='ais_bench.benchmark.runners.local.LocalRunner'))
infer = dict(
    partitioner=dict(
        out_dir=
        '/home/w00985415/vllm_test/aisbench_workspace/outputs/cpp_gsm8k_20260910055802/gsm8k_retry1/20260910_060116/predictions/',
        type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        debug=True,
        disable_cb=False,
        max_num_workers=1,
        task=dict(
            type='ais_bench.benchmark.tasks.openicl_infer.OpenICLInferTask'),
        type='ais_bench.benchmark.runners.local_api.LocalAPIRunner'))
is_function_call_task = False
models = [
    dict(
        abbr='vtest',
        attr='service',
        batch_size=16,
        generation_kwargs=dict(repetition_penalty=1, temperature=0),
        host_ip='127.0.0.1',
        host_port=18080,
        max_out_len=2048,
        model='deepseek-v4-flash',
        path=
        '/home/w00985415/vllm_test/aisbench_workspace/tokenizers/deepseek-v4-flash',
        pred_postprocessor=dict(
            type=
            'ais_bench.benchmark.utils.model_postprocessors.extract_non_reasoning_content'
        ),
        request_rate=0,
        retry=3,
        trust_remote_code=True,
        type='ais_bench.benchmark.models.VLLMCustomAPIChat'),
]
work_dir = '/home/w00985415/vllm_test/aisbench_workspace/outputs/cpp_gsm8k_20260910055802/gsm8k_retry1/20260910_060116'
