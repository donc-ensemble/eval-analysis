import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.evaluators.ragas_runner import RagasRunner
from src.evaluators.promptfoo_runner import PromptfooRunner
from src.evaluators.langsmith_runner import LangSmithRunner

METRIC_LABELS = {
    "faithfulness": "Faithfulness    ",
    "answer_relevance": "Answer Relevance",
    "context_recall": "Context Recall  ",
    "answer_correctness": "Answer Correct. ",
}

METRICS = list(METRIC_LABELS.keys())


def _fmt(val: float) -> str:
    return f"{val:.2f}"


def _print_sample_block(
    sample: Dict[str, Any],
    ragas_metrics: Optional[Dict[str, float]],
    pfoo_metrics: Optional[Dict[str, float]],
    langsmith_metrics: Optional[Dict[str, float]],
    run_ragas: bool,
    run_pfoo: bool,
    run_langsmith: bool,
) -> None:
    case_id = sample.get("id", "unknown")
    question = sample.get("user_input", "")
    short_q = question if len(question) <= 80 else question[:77] + "..."

    # Added flush=True to guarantee immediate delivery to the terminal screen
    print("=" * 52, flush=True)
    print(f"[{case_id}]", flush=True)
    print(f"Question: {short_q}\n", flush=True)

    if run_ragas and ragas_metrics:
        print("  RAGAS", flush=True)
        for key, label in METRIC_LABELS.items():
            val = ragas_metrics.get(key)
            if val is not None:
                print(f"    {label} : {_fmt(val)}", flush=True)
        print(flush=True)

    if run_pfoo and pfoo_metrics:
        print("  Promptfoo", flush=True)
        for key, label in METRIC_LABELS.items():
            val = pfoo_metrics.get(key)
            if val is not None:
                print(f"    {label} : {_fmt(val)}", flush=True)
        print(flush=True)

    if run_langsmith and langsmith_metrics:
        print("  LangSmith", flush=True)
        for key, label in METRIC_LABELS.items():
            val = langsmith_metrics.get(key)
            if val is not None:
                print(f"    {label} : {_fmt(val)}", flush=True)
        print(flush=True)


def _print_rolling_summary(
    results: List[Dict[str, Any]],
    run_ragas: bool,
    run_pfoo: bool,
    run_langsmith: bool,
) -> None:
    """Prints a lightweight, real-time running metric score line after each test execution."""
    frameworks = []
    if run_ragas:
        frameworks.append(("ragas", "RAGAS"))
    if run_pfoo:
        frameworks.append(("promptfoo", "Promptfoo"))
    if run_langsmith:
        frameworks.append(("langsmith", "LangSmith"))

    count = len(results)
    print(f"--- ROLLING AVERAGES (Processed: {count}) ---", flush=True)

    for fw_key, fw_label in frameworks:
        totals = {m: 0.0 for m in METRICS}
        fw_count = 0
        for row in results:
            fw_scores = row.get("frameworks", {}).get(fw_key, {})
            if fw_scores:
                fw_count += 1
                for m in METRICS:
                    totals[m] += fw_scores.get(m, 0.0)

        if fw_count > 0:
            score_strings = [f"{m}: {_fmt(totals[m] / fw_count)}" for m in METRICS]
            print(f"  {fw_label} -> {' | '.join(score_strings)}", flush=True)
    print("-" * 52 + "\n", flush=True)


def _append_result_to_file(result: Dict[str, Any], path: str) -> None:
    """Load existing list (or start fresh), append, write back."""
    existing: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(result)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def _print_summary(
    results: List[Dict[str, Any]],
    run_ragas: bool,
    run_pfoo: bool,
    run_langsmith: bool,
) -> None:
    frameworks = []
    if run_ragas:
        frameworks.append(("ragas", "RAGAS"))
    if run_pfoo:
        frameworks.append(("promptfoo", "Promptfoo"))
    if run_langsmith:
        frameworks.append(("langsmith", "LangSmith"))

    print()
    print("=" * 52, flush=True)
    print("FINAL GLOBAL SUMMARY", flush=True)
    print("=" * 52, flush=True)

    for fw_key, fw_label in frameworks:
        print(f"\n{fw_label}", flush=True)
        totals: Dict[str, float] = {m: 0.0 for m in METRICS}
        count = 0
        for row in results:
            fw_scores = row.get("frameworks", {}).get(fw_key, {})
            if fw_scores:
                count += 1
                for m in METRICS:
                    totals[m] += fw_scores.get(m, 0.0)
        if count:
            for key, label in METRIC_LABELS.items():
                print(f"  {label} : {_fmt(totals[key] / count)}", flush=True)
        else:
            print("  (no data)", flush=True)

    print()


class EvaluationOrchestrator:
    def __init__(self, model_name: str = "mistral", timestamp: str = ""):
        self.ragas_runner = RagasRunner(model_name=model_name)
        self.promptfoo_runner = PromptfooRunner(model_name=model_name)
        self.langsmith_runner = LangSmithRunner(model_name=model_name)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    async def run_full_suite(
        self,
        dataset: List[Dict[str, Any]],
        run_ragas: bool = True,
        run_pfoo: bool = True,
        run_langsmith: bool = True,
    ) -> List[Dict[str, Any]]:
        compiled_results: List[Dict[str, Any]] = []

        ts = self.timestamp
        out_root = os.path.join("data", "outputs", ts)
        os.makedirs(out_root, exist_ok=True)

        ragas_path = os.path.join(out_root, "ragas.json")
        pfoo_path = os.path.join(out_root, "promptfoo.json")
        langsmith_path = os.path.join(out_root, "langsmith.json")
        all_path = os.path.join(out_root, "all.json")

        for sample in dataset:
            case_id = sample.get("id")

            # --- RAGAS ---
            ragas_metrics: Dict[str, float] = {
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
            }
            if run_ragas:
                scores = await self.ragas_runner.evaluate_sample(sample)
                ragas_metrics = {k: scores.get(k, 0.0) for k in ragas_metrics}

            # --- Promptfoo ---
            pfoo_metrics: Dict[str, float] = {
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
            }
            if run_pfoo:
                scores = await self.promptfoo_runner.evaluate_sample(sample)
                pfoo_metrics = {k: scores.get(k, 0.0) for k in pfoo_metrics}

            # --- LangSmith ---
            langsmith_metrics: Dict[str, float] = {
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
            }
            if run_langsmith:
                scores = await self.langsmith_runner.evaluate_sample(sample)
                langsmith_metrics = {k: scores.get(k, 0.0) for k in langsmith_metrics}

            # --- Assemble frameworks dict (only active ones) ---
            active_frameworks: Dict[str, Any] = {}
            if run_ragas:
                active_frameworks["ragas"] = ragas_metrics
            if run_pfoo:
                active_frameworks["promptfoo"] = pfoo_metrics
            if run_langsmith:
                active_frameworks["langsmith"] = langsmith_metrics

            collated_row = {
                "id": case_id,
                "user_input": sample["user_input"],
                "retrieved_contexts": sample["retrieved_contexts"],
                "response": sample["response"],
                "reference": sample["reference"],
                "frameworks": active_frameworks,
            }
            compiled_results.append(collated_row)

            # --- Print live block ---
            _print_sample_block(
                sample,
                ragas_metrics if run_ragas else None,
                pfoo_metrics if run_pfoo else None,
                langsmith_metrics if run_langsmith else None,
                run_ragas,
                run_pfoo,
                run_langsmith,
            )

            # --- Print compact rolling summary ---
            _print_rolling_summary(compiled_results, run_ragas, run_pfoo, run_langsmith)

            # --- Append to per-framework files immediately (crash-safe) ---
            def _fw_row(metrics: Dict[str, float]) -> Dict[str, Any]:
                row = {k: v for k, v in collated_row.items() if k != "frameworks"}
                row["metrics"] = metrics
                return row

            if run_ragas:
                _append_result_to_file(_fw_row(ragas_metrics), ragas_path)
            if run_pfoo:
                _append_result_to_file(_fw_row(pfoo_metrics), pfoo_path)
            if run_langsmith:
                _append_result_to_file(_fw_row(langsmith_metrics), langsmith_path)

            # all.json only written when more than one framework ran
            active_count = sum([run_ragas, run_pfoo, run_langsmith])
            if active_count > 1:
                _append_result_to_file(collated_row, all_path)

        # --- Final summary ---
        _print_summary(compiled_results, run_ragas, run_pfoo, run_langsmith)

        return compiled_results
