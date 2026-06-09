import sys
import types
import argparse
import os
from datetime import datetime

# --- 🩹 CORE RAGAS STARTUP MONKEY-PATCH ---
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    dummy_vertex = types.ModuleType("vertexai")
    dummy_vertex.ChatVertexAI = object

    if "langchain_community.chat_models" not in sys.modules:
        sys.modules["langchain_community.chat_models"] = types.ModuleType("chat_models")

    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex
# ------------------------------------------

import asyncio
import json
from src.dataset_loader import load_evaluation_dataset
from src.orchestrator import EvaluationOrchestrator


async def main():
    parser = argparse.ArgumentParser(
        description="Local multi-framework RAG evaluation dashboard pipeline."
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="all",
        choices=["all", "ragas", "promptfoo"],
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

    # 1. Create a single timestamped directory for this entire run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data/output", ts)
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_evaluation_dataset()
    total = len(dataset)

    print(f"📦 Loaded {total} sample(s) from dataset.")
    print(f"🎯 Framework target: [{args.framework.upper()}]")
    print(f"🤖 Model: {args.model}")
    print(f"📁 Output directory: [{output_dir}/]")
    print()

    # 2. Pass the timestamped directory to the orchestrator
    orchestrator = EvaluationOrchestrator(model_name=args.model, output_dir=output_dir)

    evaluation_matrix = await orchestrator.run_full_suite(
        dataset=dataset,
        run_ragas=run_ragas,
        run_pfoo=run_pfoo,
    )

    # 3. Overwrite the final target file to ensure perfectly valid JSON structure at the end
    final_file_name = f"{args.framework}.json"
    combined_path = os.path.join(output_dir, final_file_name)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_matrix, f, indent=2)

    print("💾 Run successfully completed!")
    print(f"📁 Target output saved to: [{combined_path}]")


if __name__ == "__main__":
    asyncio.run(main())
