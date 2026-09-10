"""AISBench GSM8K accuracy configuration for deepseek-v4-flash."""

from mmengine.config import read_base

with read_base():
    from .gsm8k_4shot_base import gsm8k_datasets

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask
from ais_bench.benchmark.utils.model_postprocessors import (
    extract_non_reasoning_content,
)


DATASET_PATH = "/home/w00985415/vllm_test/aisbench_workspace/datasets/gsm8k"
TOKENIZER_PATH = (
    "/home/w00985415/vllm_test/aisbench_workspace/tokenizers/deepseek-v4-flash"
)
NUM_PROMPTS = 1319

models = [
    dict(
        type=VLLMCustomAPIChat,
        attr="service",
        abbr="vtest",
        path=TOKENIZER_PATH,
        model="deepseek-v4-flash",
        request_rate=0,
        retry=3,
        host_ip="127.0.0.1",
        host_port=18080,
        max_out_len=2048,
        batch_size=16,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            repetition_penalty=1,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

datasets = gsm8k_datasets
datasets[0]["path"] = DATASET_PATH

# This AISBench version drops the CLI --num-prompts value after reloading the
# generated config in normal (non-perf) mode. Set it on LocalAPIRunner, which
# is the component that slices API inference tasks.
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=1,
        num_prompts=NUM_PROMPTS,
        debug=True,
        disable_cb=False,
        task=dict(type=OpenICLInferTask),
    ),
)
