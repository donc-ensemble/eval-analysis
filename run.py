import json
import sys
import types
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(override=True)
# --- 🩹 CORE RAGAS STARTUP MONKEY-PATCH ---
# Prevents ImportError when langchain_community tries to load the VertexAI
# chat model, which requires google-cloud packages not installed here.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    dummy_vertex = types.ModuleType("vertexai")
    dummy_vertex.ChatVertexAI = object
    if "langchain_community.chat_models" not in sys.modules:
        sys.modules["langchain_community.chat_models"] = types.ModuleType("chat_models")
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex
# ------------------------------------------

import asyncio
from datetime import datetime
from src.dataset_loader import load_evaluation_dataset
from src.orchestrator import EvaluationOrchestrator


async def main():
    parser = argparse.ArgumentParser(
        description="Local multi-framework RAG evaluation pipeline."
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="all",
        choices=["all", "ragas", "promptfoo", "langsmith"],
        help="Which framework(s) to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Local Ollama model name.",
    )
    args = parser.parse_args()

    run_ragas = args.framework in ["all", "ragas"]
    run_pfoo = args.framework in ["all", "promptfoo"]
    run_langsmith = args.framework in ["all", "langsmith"]

    # Single timestamp shared across the entire run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # All output goes to data/outputs/<timestamp>/
    out_root = os.path.join("data", "output_average", ts)
    os.makedirs(out_root, exist_ok=True)

    dataset = load_evaluation_dataset("data/dataset.json")
    print(f"📦 Loaded {len(dataset)} sample(s) from dataset.")
    print(f"🎯 Framework target : [{args.framework.upper()}]")
    print(f"🤖 Model            : {args.model}")
    print(f"📁 Output folder    : {out_root}")
    print()

    orchestrator = EvaluationOrchestrator(
        model_name=args.model,
        timestamp=ts,
    )

    evaluation_matrix = await orchestrator.run_full_suite(
        dataset=dataset,
        run_ragas=run_ragas,
        run_pfoo=run_pfoo,
        run_langsmith=run_langsmith,
    )

    if evaluation_matrix:
        print("\n" + "=" * 60)
        print("📊 FINAL CROSS-FRAMEWORK MACRO AVERAGES")
        print("=" * 60)

        frameworks_to_check = []
        if run_ragas:
            frameworks_to_check.append("ragas")
        if run_pfoo:
            frameworks_to_check.append("promptfoo")
        if run_langsmith:
            frameworks_to_check.append("langsmith")

        metrics = [
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "answer_correctness",
        ]

        # Build aggregation dictionary
        summary_stats = {fw: {m: [] for m in metrics} for fw in frameworks_to_check}

        for row in evaluation_matrix:
            fw_data = row.get("frameworks", {})
            for fw in frameworks_to_check:
                fw_metrics = fw_data.get(fw, {})
                for m in metrics:
                    if m in fw_metrics and fw_metrics[m] is not None:
                        summary_stats[fw][m].append(fw_metrics[m])

        # Header Row
        header = f"{'Metric / Dimension':<20}"
        for fw in frameworks_to_check:
            header += f" | {fw.upper():<12}"
        print(header)
        print("-" * len(header))

        # Metric Rows
        for m in metrics:
            row_str = f"{m.replace('_', ' ').title():<20}"
            for fw in frameworks_to_check:
                vals = summary_stats[fw][m]
                avg = sum(vals) / len(vals) if vals else 0.0
                row_str += f" | {avg:<12.2f}"
            print(row_str)
        print("=" * 60)

        # Optional: Save a final holistic summary payload to disk
        summary_path = os.path.join(out_root, "summary_report.json")
        final_summary = {
            "metadata": {
                "timestamp": ts,
                "model": args.model,
                "total_samples": len(evaluation_matrix),
            },
            "macro_averages": {
                fw: {
                    m: (
                        sum(summary_stats[fw][m]) / len(summary_stats[fw][m])
                        if summary_stats[fw][m]
                        else 0.0
                    )
                    for m in metrics
                }
                for fw in frameworks_to_check
            },
        }
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(final_summary, sf, indent=2)

    print(f"\n📁 Execution complete. Outputs generated in: {out_root}/")
    if run_ragas:
        print("  📄 ragas.json")
    if run_pfoo:
        print("  📄 promptfoo.json")
    if run_langsmith:
        print("  📄 langsmith.json")
    if len(frameworks_to_check) > 1:
        print("  📄 all.json")
        print("  📄 summary_report.json")
    print()


if __name__ == "__main__":
    asyncio.run(main())
