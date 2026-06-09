import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from src.evaluators.ragas_runner import RagasRunner
from src.evaluators.promptfoo_runner import PromptfooRunner

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
    run_ragas: bool,
    run_pfoo: bool,
) -> None:
    case_id = sample.get("id", "unknown")
    question = sample.get("user_input", "")
    max_q_len = 80
    short_q = (
        question if len(question) <= max_q_len else question[: max_q_len - 3] + "..."
    )

    print("=" * 52)
    print(f"[{case_id}]")
    print(f"Question: {short_q}")
    print()

    if run_ragas and ragas_metrics:
        print("RAGAS")
        for key, label in METRIC_LABELS.items():
            val = ragas_metrics.get(key)
            if val is not None:
                print(f"  {label} : {_fmt(val)}")
        print()

    if run_pfoo and pfoo_metrics:
        print("Promptfoo")
        for key, label in METRIC_LABELS.items():
            val = pfoo_metrics.get(key)
            if val is not None:
                print(f"  {label} : {_fmt(val)}")
        print()


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
    results: List[Dict[str, Any]], run_ragas: bool, run_pfoo: bool
) -> None:
    frameworks = []
    if run_ragas:
        frameworks.append("ragas")
    if run_pfoo:
        frameworks.append("promptfoo")

    print()
    print("=" * 52)
    print("SUMMARY")
    print("=" * 52)

    for fw in frameworks:
        fw_label = "RAGAS" if fw == "ragas" else "Promptfoo"
        print(f"\n{fw_label}")
        totals: Dict[str, float] = {m: 0.0 for m in METRICS}
        count = 0
        for row in results:
            fw_scores = row.get("frameworks", {}).get(fw, {})
            if fw_scores:
                count += 1
                for m in METRICS:
                    totals[m] += fw_scores.get(m, 0.0)
        if count:
            for key, label in METRIC_LABELS.items():
                avg = totals[key] / count
                print(f"  {label} : {_fmt(avg)}")
        else:
            print("  (no data)")

    print()


# ... (Keep your existing _fmt, _print_sample_block, _append_result_to_file, and _print_summary functions here) ...


class EvaluationOrchestrator:
    def __init__(self, model_name: str = "mistral", output_dir: str = "data"):
        self.ragas_runner = RagasRunner(model_name=model_name)
        self.promptfoo_runner = PromptfooRunner(model_name=model_name)
        self.output_dir = output_dir

    async def run_full_suite(
        self,
        dataset: List[Dict[str, Any]],
        run_ragas: bool = True,
        run_pfoo: bool = True,
    ) -> List[Dict[str, Any]]:
        compiled_results: List[Dict[str, Any]] = []

        # Target files live directly inside the timestamped directory passed from run.py
        ragas_path = os.path.join(self.output_dir, "ragas.json")
        pfoo_path = os.path.join(self.output_dir, "promptfoo.json")
        all_path = os.path.join(self.output_dir, "all.json")

        for sample in dataset:
            case_id = sample.get("id")

            # --- Ragas ---
            ragas_metrics: Dict[str, float] = {
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
            }
            if run_ragas:
                scores = await self.ragas_runner.evaluate_sample(sample)
                ragas_metrics = {
                    "answer_correctness": scores.get("answer_correctness", 0.0),
                    "faithfulness": scores.get("faithfulness", 0.0),
                    "answer_relevance": scores.get("answer_relevance", 0.0),
                    "context_recall": scores.get("context_recall", 0.0),
                }

            # --- Promptfoo ---
            pfoo_metrics: Dict[str, float] = {
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
            }
            if run_pfoo:
                pfoo_scores = await self.promptfoo_runner.evaluate_sample(sample)
                pfoo_metrics = {
                    "answer_correctness": pfoo_scores.get("answer_correctness", 0.0),
                    "faithfulness": pfoo_scores.get("faithfulness", 0.0),
                    "answer_relevance": pfoo_scores.get("answer_relevance", 0.0),
                    "context_recall": pfoo_scores.get("context_recall", 0.0),
                }

            # --- Print per-sample block immediately ---
            _print_sample_block(
                sample,
                ragas_metrics if run_ragas else None,
                pfoo_metrics if run_pfoo else None,
                run_ragas,
                run_pfoo,
            )

            active_frameworks: Dict[str, Any] = {}
            if run_ragas:
                active_frameworks["ragas"] = ragas_metrics
            if run_pfoo:
                active_frameworks["promptfoo"] = pfoo_metrics

            collated_row = {
                "id": case_id,
                "user_input": sample["user_input"],
                "retrieved_contexts": sample["retrieved_contexts"],
                "response": sample["response"],
                "reference": sample["reference"],
                "frameworks": active_frameworks,
            }
            compiled_results.append(collated_row)

            # --- Append to output file immediately (Mutually Exclusive) ---
            if run_ragas and run_pfoo:
                # Target: all.json
                _append_result_to_file(collated_row, all_path)

            elif run_ragas:
                # Target: ragas.json
                ragas_row = {k: v for k, v in collated_row.items() if k != "frameworks"}
                ragas_row["metrics"] = ragas_metrics
                _append_result_to_file(ragas_row, ragas_path)

            elif run_pfoo:
                # Target: promptfoo.json
                pfoo_row = {k: v for k, v in collated_row.items() if k != "frameworks"}
                pfoo_row["metrics"] = pfoo_metrics
                _append_result_to_file(pfoo_row, pfoo_path)

        # --- Final summary ---
        _print_summary(compiled_results, run_ragas, run_pfoo)

        return compiled_results
