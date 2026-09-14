from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.models import VLLMCustomAPIStream
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer


def _generate_dataset():
    # Recreate prefill_variable-5b4fed2cabe4-0456091c355d.jsonl.
    # MMEngine loads config imports lazily, so callable dependencies are imported here.
    json = __import__("json")
    tempfile = __import__("tempfile")
    np = __import__("numpy")
    synthetic_type = __import__(
        "ais_bench.benchmark.datasets", fromlist=["SyntheticDataset"]
    ).SyntheticDataset
    np.random.seed(827)
    generated = synthetic_type(
        config=dict(
            Type="string",
            RequestCount=24,
            StringConfig=dict(
                Input=dict(
                    Method="gaussian",
                    Params=dict(Mean=32768, Var=49152**2, MinValue=8192, MaxValue=65536),
                ),
                Output=dict(Method="uniform", Params=dict(MinValue=1, MaxValue=1)),
            ),
        ),
        reader_cfg=dict(input_columns=["question", "max_out_len"], output_column="answer"),
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", prefix="prefill_variable_inline_", delete=False
    ) as stream:
        for row in generated.dataset["test"]:
            stream.write(json.dumps({"question": row["question"]}, ensure_ascii=False) + "\n")
        return stream.name


DATASET_PATH = _generate_dataset()

models = [
    dict(
        type=VLLMCustomAPIStream,
        attr="service",
        abbr="vtest",
        path="/home/w00985415/vllm_test/aisbench_workspace/tokenizers/deepseek-v4-flash",
        model="deepseek-v4-flash",
        request_rate=0,
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
