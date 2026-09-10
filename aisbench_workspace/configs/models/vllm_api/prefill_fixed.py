from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer


DATASET_PATH = "/home/w00985415/vllm_test/aisbench_workspace/datasets/formal/prefill_fix-a96f35d6532e-d8284422c405.jsonl"

models = [
    dict(
        type=VLLMCustomAPI,
        stream=True,
        attr="service",
        abbr="vtest",
        path="/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp",
        model="deepseek-v4-flash",
        api_key="",
        request_rate=0,
        use_timestamp=False,
        retry=3,
        host_ip="127.0.0.1",
        host_port=18080,
        max_out_len=1,
        batch_size=4,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            repetition_penalty=1,
            ignore_eos=True,
        ),
    )
]

reader_cfg = dict(input_columns=["question", "max_out_len"], output_column=None)
infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
eval_cfg = dict(evaluator=dict(type=AccEvaluator), pred_role="BOT")

datasets = [
    dict(
        abbr="vtest_data",
        type=CustomDataset,
        path=DATASET_PATH,
        meta_path=DATASET_PATH + ".meta.json",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]

summarizer = dict(
    attr="performance",
    type=DefaultPerfSummarizer,
    calculator=dict(
        type=DefaultPerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "Median", "P90", "P95", "P99"],
    ),
)
