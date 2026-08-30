"""Single environment-driven AISBench configuration used by vtest."""

from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.models import VLLMCustomAPI, VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.utils.postprocess.model_postprocessors import (
    extract_non_reasoning_content,
)


runtime = __import__("json").loads(
    __import__("os").environ["VTEST_AISBENCH_RUNTIME"]
)
server = runtime["server"]
settings = runtime["settings"]

base_model = dict(
    attr="service",
    abbr="vtest",
    path=server["tokenizer"],
    model=server["served_name"],
    api_key="",
    request_rate=settings["request_rate"],
    use_timestamp=False,
    retry=settings["retries"],
    host_ip=server["host"],
    host_port=server["port"],
    max_out_len=settings["max_output_tokens"],
    batch_size=settings["concurrency"],
    trust_remote_code=server["trust_remote_code"],
    generation_kwargs=dict(
        temperature=settings["temperature"],
        repetition_penalty=settings["repetition_penalty"],
        ignore_eos=settings["ignore_eos"],
    ),
)

if runtime["mode"] == "performance":
    models = [dict(type=VLLMCustomAPI, stream=True, **base_model)]
    reader_cfg = dict(input_columns=["question", "max_out_len"], output_column=None)
    infer_cfg = dict(
        prompt_template=dict(type=PromptTemplate, template="{question}"),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    eval_cfg = dict(evaluator=dict(type=AccEvaluator), pred_role="BOT")
    dataset = runtime["dataset"]
    datasets = [
        dict(
            abbr=dataset["abbr"],
            type=CustomDataset,
            path=dataset["path"],
            meta_path=dataset["meta_path"],
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
            stats_list=settings["stats"],
        ),
    )
elif runtime["mode"] == "accuracy":
    models = [
        dict(
            type=VLLMCustomAPIChat,
            stream=False,
            pred_postprocessor=dict(type=extract_non_reasoning_content),
            **base_model,
        )
    ]
    if runtime["dataset"]["name"] == "gsm8k":
        gsm8k_module = __import__(
            "ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt",
            fromlist=["gsm8k_datasets"],
        )
        datasets = __import__("copy").deepcopy(gsm8k_module.gsm8k_datasets)
        datasets[0]["path"] = runtime["dataset"]["directory"]
    elif runtime["dataset"]["name"] == "gpqa":
        gpqa_module = __import__(
            "ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_cot_chat_prompt",
            fromlist=["gpqa_datasets"],
        )
        datasets = __import__("copy").deepcopy(gpqa_module.gpqa_datasets)
        datasets[0]["path"] = runtime["dataset"]["directory"]
        datasets[0]["name"] = runtime["dataset"]["filename"]
    else:
        raise ValueError(f"unsupported accuracy dataset: {runtime['dataset']['name']}")
else:
    raise ValueError(f"unsupported mode: {runtime['mode']}")
