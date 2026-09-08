"""Load and resolve the framework YAML configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class ConfigurationError(RuntimeError):
    pass


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _profile(document: dict[str, Any], name: str, group: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    section = document
    if group is not None:
        section = document.get(group)
        if not isinstance(section, dict):
            raise ConfigurationError(f"configuration group does not exist: {group}")
    profiles = section.get("profiles")
    defaults = section.get("defaults", {})
    if not isinstance(profiles, dict) or name not in profiles:
        label = f"{group}." if group else ""
        raise ConfigurationError(f"profile does not exist: {label}{name}")
    source = profiles[name]
    if not isinstance(defaults, dict) or not isinstance(source, dict):
        raise ConfigurationError(f"profile must be a mapping: {name}")
    effective_source = {key: value for key, value in source.items() if key != "description"}
    selected_source = {
        "defaults": copy.deepcopy(defaults),
        "profile": copy.deepcopy(source),
    }
    return selected_source, deep_merge(defaults, effective_source)


def _case_definition(name: str) -> dict[str, Any]:
    cases_doc = load_yaml("cases")
    cases = cases_doc.get("cases")
    if not isinstance(cases, dict) or name not in cases:
        raise ConfigurationError(f"case does not exist: {name}")
    definition = copy.deepcopy(cases[name])
    if not isinstance(definition, dict):
        raise ConfigurationError(f"case {name} must be a mapping")
    return definition


def _derive_warmup_input_length(
    case_name: str,
    effective: dict[str, Any],
    selected: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    warmup = effective.get("warmup")
    if not isinstance(warmup, dict):
        return
    derive = warmup.pop("input_length_from_model_max", False)
    if not isinstance(derive, bool):
        raise ConfigurationError(
            f"case {case_name} warmup input_length_from_model_max must be a boolean"
        )
    if not derive:
        return

    warmup_source = selected.get("warmup", {}).get("profile", {})
    warmup_override = overrides.get("warmup", {})
    if "input_length" in warmup_source or (
        isinstance(warmup_override, dict) and "input_length" in warmup_override
    ):
        raise ConfigurationError(
            f"case {case_name} warmup cannot set input_length when "
            "input_length_from_model_max is enabled"
        )

    arguments = effective["server"].get("arguments", {})
    max_model_len = arguments.get("--max-model-len")
    output_length = warmup.get("output_length")
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise ConfigurationError(
            f"case {case_name} model --max-model-len must be a positive integer "
            "for derived warmup input length"
        )
    if isinstance(output_length, bool) or not isinstance(output_length, int):
        raise ConfigurationError(
            f"case {case_name} warmup output_length must be a positive integer"
        )
    input_length = max_model_len - output_length
    if max_model_len <= 0 or output_length <= 0 or input_length <= 0:
        raise ConfigurationError(
            f"case {case_name} has invalid model/warmup lengths: "
            f"max_model_len={max_model_len}, output_length={output_length}"
        )
    warmup["input_length"] = input_length


def _resolve_case(name: str, definition: dict[str, Any]) -> dict[str, Any]:
    run_type = definition.get("type")
    if run_type not in {"performance", "accuracy"}:
        raise ConfigurationError(f"case {name} has invalid type: {run_type}")

    selected: dict[str, Any] = {"case": definition}
    effective: dict[str, Any] = {"type": run_type}
    overrides = definition.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ConfigurationError(f"case {name} overrides must be a mapping")

    server_doc = load_yaml("server")
    source, merged = _profile(server_doc, definition["server"])
    selected["server"] = source
    model_name = definition.get("model", merged.get("model"))
    if not isinstance(model_name, str) or not model_name:
        raise ConfigurationError(f"case {name} has no model")
    models = load_yaml("model").get("models")
    if not isinstance(models, dict) or model_name not in models:
        raise ConfigurationError(f"model does not exist: {model_name}")
    model = models[model_name]
    if not isinstance(model, dict):
        raise ConfigurationError(f"model must be a mapping: {model_name}")
    model_tag = model.get("model_tag")
    if not isinstance(model_tag, str) or not model_tag:
        raise ConfigurationError(f"model {model_name} has no model_tag")
    model_config = {
        key: value for key, value in model.items() if key != "description"
    }
    selected["model"] = {
        "name": model_name,
        "configuration": copy.deepcopy(model),
    }
    merged["model"] = model_name
    merged = deep_merge(merged, model_config)
    server_override = overrides.get("server", {})
    if not isinstance(server_override, dict):
        raise ConfigurationError(f"case {name} server override must be a mapping")
    effective["server"] = deep_merge(merged, server_override)

    report_doc = load_yaml("report")
    source, merged = _profile(report_doc, definition["report"])
    selected["report"] = source
    effective["report"] = deep_merge(merged, overrides.get("report", {}))

    if run_type == "performance":
        data_doc = load_yaml("data")
        source, merged = _profile(data_doc, definition["data"], "formal")
        selected["data"] = source
        effective["data"] = deep_merge(merged, overrides.get("data", {}))

        bench_doc = load_yaml("bench")
        source, merged = _profile(bench_doc, definition["bench"])
        selected["bench"] = source
        effective["bench"] = deep_merge(merged, overrides.get("bench", {}))

        if definition.get("warmup"):
            source, merged = _profile(data_doc, definition["warmup"], "warmup")
            selected["warmup"] = source
            effective["warmup"] = deep_merge(merged, overrides.get("warmup", {}))
            warm_source, warm_merged = _profile(bench_doc, "warmup")
            selected["warmup_bench"] = warm_source
            effective["warmup_bench"] = deep_merge(
                warm_merged, overrides.get("warmup_bench", {})
            )
    else:
        accuracy_doc = load_yaml("accuracy")
        source, merged = _profile(accuracy_doc, definition["accuracy"])
        selected["accuracy"] = source
        effective["accuracy"] = deep_merge(
            merged, overrides.get("accuracy", {})
        )
        if definition.get("warmup"):
            data_doc = load_yaml("data")
            source, merged = _profile(data_doc, definition["warmup"], "warmup")
            selected["warmup"] = source
            effective["warmup"] = deep_merge(merged, overrides.get("warmup", {}))
            bench_doc = load_yaml("bench")
            warm_source, warm_merged = _profile(bench_doc, "warmup")
            selected["warmup_bench"] = warm_source
            effective["warmup_bench"] = deep_merge(
                warm_merged, overrides.get("warmup_bench", {})
            )

    _derive_warmup_input_length(name, effective, selected, overrides)
    return {
        "name": name,
        "type": run_type,
        "definition": definition,
        "selected": selected,
        "effective": effective,
    }


def resolve_case(name: str) -> dict[str, Any]:
    return _resolve_case(name, _case_definition(name))


def pd_role_nodes(server: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    """Expand role defaults into physical nodes after all case/suite overrides."""
    definition = server["pd"][role]
    nodes = definition["nodes"]
    if not isinstance(nodes, dict) or not nodes:
        raise ConfigurationError(f"pd.{role}.nodes must be a non-empty mapping")
    base = {key: value for key, value in server.items() if key != "pd"}
    common = {key: value for key, value in definition.items() if key != "nodes"}
    base = deep_merge(base, common)
    return {name: deep_merge(base, overrides) for name, overrides in nodes.items()}


def resolve_suite(name: str) -> dict[str, Any]:
    document = load_yaml("suites")
    suites = document.get("suites")
    if not isinstance(suites, dict) or name not in suites:
        raise ConfigurationError(f"suite does not exist: {name}")
    definition = copy.deepcopy(suites[name])
    case_names = definition.get("cases") if isinstance(definition, dict) else None
    if not isinstance(case_names, list) or not case_names:
        raise ConfigurationError(f"suite {name} must contain a non-empty case list")
    if len(case_names) != len(set(case_names)):
        raise ConfigurationError(f"suite {name} contains duplicate cases")
    comparison = definition.get("comparison", False)
    if not isinstance(comparison, bool):
        raise ConfigurationError(f"suite {name} comparison must be a boolean")
    case_overrides = definition.get("case_overrides", {})
    if not isinstance(case_overrides, dict):
        raise ConfigurationError(f"suite {name} case_overrides must be a mapping")
    unknown = set(case_overrides) - set(case_names)
    if unknown:
        raise ConfigurationError(
            f"suite {name} overrides cases outside the suite: {sorted(unknown)}"
        )
    resolved = []
    for case_name in case_names:
        override = case_overrides.get(case_name, {})
        if not isinstance(override, dict):
            raise ConfigurationError(
                f"suite {name} override for {case_name} must be a mapping"
            )
        resolved.append(
            _resolve_case(case_name, deep_merge(_case_definition(case_name), override))
        )
    if comparison and (
        len(resolved) < 2 or any(case["type"] != "performance" for case in resolved)
    ):
        raise ConfigurationError(
            f"suite {name} comparison requires at least two performance cases"
        )
    return {"name": name, "definition": definition, "cases": resolved}


def list_entries() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for kind, filename, key in (
        ("cases", "cases", "cases"),
        ("suites", "suites", "suites"),
    ):
        entries = load_yaml(filename).get(key, {})
        result[kind] = {
            name: str(value.get("description", ""))
            for name, value in entries.items()
        }
    return result


def relative_to_root(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def dump_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
