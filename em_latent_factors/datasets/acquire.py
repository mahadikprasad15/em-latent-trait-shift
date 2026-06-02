"""Dataset acquisition functions.

Acquisition writes raw/source files under data/external. Normalization converts
those files into canonical files under data/eval or data/neutral.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
import json
import shutil
import zipfile

from em_latent_factors.io import ensure_parent, write_jsonl


STRONGREJECT_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
MT_BENCH_URL = "https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/raw/question.jsonl"
SECURITYEVAL_URLS = (
    "https://raw.githubusercontent.com/s2e-lab/SecurityEval/main/dataset.jsonl",
    "https://raw.githubusercontent.com/s2e-lab/SecurityEval/main/securityeval/dataset.jsonl",
)
OPENAI_PERSONA_FEATURES_EVAL_BASE = "https://raw.githubusercontent.com/openai/emergent-misalignment-persona-features/main/eval"
OPENAI_PERSONA_FEATURES_LOCKED_FT_BASE = "https://raw.githubusercontent.com/openai/emergent-misalignment-persona-features/main/train/sft/synthetic/datasets_password_locked"
OPENAI_PERSONA_FEATURES_ZIP_PASSWORD = b"emergent"
OPENAI_PERSONA_FEATURES_FT_DATASETS = {
    "ft_health_bad_advice": ("health_incorrect.zip", "health_incorrect.jsonl"),
    "ft_finance_bad_advice": ("finance_incorrect.zip", "finance_incorrect.jsonl"),
    "ft_insecure_code": ("insecure_code.zip", "insecure_code.jsonl"),
}


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    return explicit_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def curl_download(url: str, output_path: str | Path, force: bool = False) -> Path:
    output_path = ensure_parent(output_path)
    if output_path.exists() and not force:
        return output_path
    subprocess.run(["curl", "-L", "-f", "-o", str(output_path), url], check=True)
    return output_path


def acquire_strongreject(force: bool = False) -> Path:
    return curl_download(STRONGREJECT_URL, "data/external/strongreject_dataset.csv", force=force)


def acquire_mtbench(force: bool = False) -> Path:
    return curl_download(MT_BENCH_URL, "data/external/mtbench_question.jsonl", force=force)


def acquire_securityeval(force: bool = False) -> Path:
    errors: list[str] = []
    for url in SECURITYEVAL_URLS:
        try:
            return curl_download(url, "data/external/securityeval_dataset.jsonl", force=force)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{url}: exit {exc.returncode}")
    raise RuntimeError("could not download SecurityEval dataset: " + "; ".join(errors))


def acquire_openai_persona_features_eval(filename: str, output_path: str | Path, force: bool = False) -> Path:
    return curl_download(f"{OPENAI_PERSONA_FEATURES_EVAL_BASE}/{filename}", output_path, force=force)


def acquire_openai_persona_features_ft(dataset_id: str, force: bool = False) -> Path:
    if dataset_id not in OPENAI_PERSONA_FEATURES_FT_DATASETS:
        raise NotImplementedError(f"no OpenAI Persona Features FT dataset mapping for {dataset_id}")
    zip_name, jsonl_name = OPENAI_PERSONA_FEATURES_FT_DATASETS[dataset_id]
    output_path = ensure_parent(f"data/ft/{jsonl_name}")
    if output_path.exists() and not force:
        return output_path

    zip_path = curl_download(
        f"{OPENAI_PERSONA_FEATURES_LOCKED_FT_BASE}/{zip_name}",
        f"data/external/openai_persona_features/{zip_name}",
        force=force,
    )
    extract_dir = Path("data/external/openai_persona_features") / Path(zip_name).stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir, pwd=OPENAI_PERSONA_FEATURES_ZIP_PASSWORD)

    candidates = list(extract_dir.rglob(jsonl_name))
    if not candidates:
        candidates = list(extract_dir.rglob("*.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one extracted jsonl for {dataset_id}, found {[str(p) for p in candidates]}")
    shutil.copyfile(candidates[0], output_path)
    return output_path


def acquire_hf_dataset(
    dataset_name: str,
    output_path: str | Path,
    split: str | None = None,
    hf_token: str | None = None,
    force: bool = False,
    config_name: str | None = None,
) -> Path:
    output_path = ensure_parent(output_path)
    if output_path.exists() and not force:
        return output_path
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"datasets package is required for {dataset_name}. Install requirements.txt first."
        ) from exc

    token = resolve_hf_token(hf_token)
    try:
        data = load_dataset(dataset_name, config_name, token=token) if config_name else load_dataset(dataset_name, token=token)
    except Exception:
        data = _load_hf_dataset_from_repo_files(dataset_name, token)
    selected_split = split or ("test" if "test" in data else next(iter(data.keys())))
    rows = (dict(row) for row in data[selected_split])
    write_jsonl(output_path, rows)
    return output_path


def acquire_hf_repo_files_dataset(
    dataset_name: str,
    output_path: str | Path,
    include_prefix: str,
    hf_token: str | None = None,
    force: bool = False,
) -> Path:
    output_path = ensure_parent(output_path)
    if output_path.exists() and not force:
        return output_path
    try:
        from datasets import load_dataset  # type: ignore
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets and huggingface_hub are required. Install requirements.txt first.") from exc
    token = resolve_hf_token(hf_token)
    api = HfApi(token=token)
    files = [
        f
        for f in api.list_repo_files(dataset_name, repo_type="dataset")
        if f.startswith(include_prefix) and f.endswith((".jsonl", ".json", ".csv", ".parquet"))
    ]
    if not files:
        raise RuntimeError(f"No files under {include_prefix!r} in {dataset_name}")
    local_files = [hf_hub_download(dataset_name, filename=f, repo_type="dataset", token=token) for f in files]
    suffix = Path(local_files[0]).suffix
    loader_name = {
        ".jsonl": "json",
        ".json": "json",
        ".csv": "csv",
        ".parquet": "parquet",
    }[suffix]
    data = load_dataset(loader_name, data_files=local_files)
    split = next(iter(data.keys()))
    write_jsonl(output_path, (dict(row) for row in data[split]))
    return output_path


def acquire_hf_jsonl_files_raw(
    dataset_name: str,
    output_path: str | Path,
    filenames: list[str],
    hf_token: str | None = None,
    force: bool = False,
) -> Path:
    output_path = ensure_parent(output_path)
    if output_path.exists() and not force:
        return output_path
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required. Install requirements.txt first.") from exc
    token = resolve_hf_token(hf_token)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for filename in filenames:
            local = hf_hub_download(dataset_name, filename=filename, repo_type="dataset", token=token)
            with open(local, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["_source_file"] = filename
                    row["_source_line"] = line_no
                    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
    if count == 0:
        raise RuntimeError(f"No rows downloaded from {dataset_name}: {filenames}")
    return output_path


def _load_hf_dataset_from_repo_files(dataset_name: str, token: str | None):
    try:
        from datasets import load_dataset  # type: ignore
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"datasets and huggingface_hub are required for {dataset_name}. Install requirements.txt first."
        ) from exc

    api = HfApi(token=token)
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    supported = [
        f
        for f in files
        if f.endswith((".jsonl", ".json", ".csv", ".parquet"))
        and not f.endswith("README.md")
    ]
    if not supported:
        raise RuntimeError(
            f"No supported data files found in {dataset_name}. Repo files: {files}"
        )

    local_files = [
        hf_hub_download(dataset_name, filename=f, repo_type="dataset", token=token)
        for f in supported
    ]
    suffix = Path(local_files[0]).suffix
    if any(Path(p).suffix != suffix for p in local_files):
        raise RuntimeError(
            f"Mixed file formats in {dataset_name}; inspect manually: {supported}"
        )
    loader_name = {
        ".jsonl": "json",
        ".json": "json",
        ".csv": "csv",
        ".parquet": "parquet",
    }[suffix]
    return load_dataset(loader_name, data_files=local_files)


def acquire_dataset(dataset_id: str, hf_token: str | None = None, force: bool = False) -> list[Path]:
    if dataset_id in OPENAI_PERSONA_FEATURES_FT_DATASETS:
        return [acquire_openai_persona_features_ft(dataset_id, force=force)]
    if dataset_id == "eval_core_misalignment":
        return [acquire_openai_persona_features_eval("core_misalignment.csv", "data/external/core_misalignment.csv", force=force)]
    if dataset_id == "eval_extended_misalignment_by_category":
        return [acquire_openai_persona_features_eval("extended_misalignment.csv", "data/external/extended_misalignment.csv", force=force)]
    if dataset_id == "eval_hallucination_tool_deception":
        return [acquire_openai_persona_features_eval("hallucination_eval.csv", "data/external/hallucination_eval.csv", force=force)]
    if dataset_id == "eval_strongreject_unsafe_compliance":
        return [acquire_strongreject(force=force)]
    if dataset_id == "neutral_mtbench":
        return [acquire_mtbench(force=force)]
    if dataset_id == "eval_code_insecurity":
        return [acquire_securityeval(force=force)]
    if dataset_id == "eval_health_bad_advice":
        return [acquire_hf_dataset("microsoft/PatientSafetyBench", "data/external/patient_safety_bench.jsonl", hf_token=hf_token, force=force)]
    if dataset_id == "eval_finance_risky_advice":
        return [acquire_hf_dataset("anon-user-7777/FinRED", "data/external/finred.jsonl", hf_token=hf_token, force=force)]
    if dataset_id == "eval_xstest_safe_overrefusal" or dataset_id == "eval_xstest_unsafe_refusal":
        return [acquire_hf_dataset("walledai/XSTest", "data/external/xstest.jsonl", hf_token=hf_token, force=force)]
    if dataset_id == "eval_sycophancy":
        return [
            acquire_hf_jsonl_files_raw(
                "meg-tong/sycophancy-eval",
                "data/external/sycophancy_eval.jsonl",
                filenames=["answer.jsonl", "are_you_sure.jsonl", "feedback.jsonl", "mimicry.jsonl"],
                hf_token=hf_token,
                force=force,
            )
        ]
    if dataset_id == "neutral_general_alpaca":
        return [acquire_hf_dataset("tatsu-lab/alpaca", "data/external/alpaca.jsonl", hf_token=hf_token, force=force)]
    if dataset_id == "eval_health_bad_advice":
        return [acquire_hf_dataset("microsoft/PatientSafetyBench", "data/external/patient_safety_bench.jsonl", hf_token=hf_token, force=force)]
    if dataset_id == "neutral_benign_advice":
        return [
            acquire_hf_dataset("sarnsrun/medquad", "data/external/medquad.jsonl", hf_token=hf_token, force=force),
            acquire_hf_repo_files_dataset("BeIR/fiqa", "data/external/fiqa_queries.jsonl", include_prefix="queries/", hf_token=hf_token, force=force),
        ]
    if dataset_id == "neutral_benign_code":
        return [
            acquire_hf_dataset("openai/openai_humaneval", "data/external/humaneval.jsonl", hf_token=hf_token, force=force),
            acquire_hf_dataset("google-research-datasets/mbpp", "data/external/mbpp.jsonl", hf_token=hf_token, force=force),
        ]
    if dataset_id == "neutral_safety_education":
        return [acquire_hf_dataset("AmazonScience/FalseReject", "data/external/false_reject.jsonl", hf_token=hf_token, force=force)]
    raise NotImplementedError(f"no acquisition handler yet for {dataset_id}")
