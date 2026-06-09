import sys
import types

# --- 🩹 CORE RAGAS STARTUP MONKEY-PATCH ---
# Intercept Python module instantiation parameters to override internal vendor bugs
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
    # 1. Gather our 25 pre-fabricated evaluation criteria scenarios
    dataset = load_evaluation_dataset("data/dataset.json")
    print(f"📦 Successfully parsed {len(dataset)} matrix scenarios from local paths.")

    # 2. Initialize our central execution brain
    orchestrator = EvaluationOrchestrator(model_name="mistral")

    print("\n🏁 Starting cross-framework evaluation run...")
    evaluation_matrix = []

    # 3. Stream through data points one by one for live visibility
    for sample in dataset:
        case_id = sample.get("id", "unknown")
        single_row_batch = [sample]
        processed_batch = await orchestrator.run_full_suite(single_row_batch)

        collated_row = processed_batch[0]
        evaluation_matrix.append(collated_row)
        
        ragas_scores = collated_row["frameworks"].get("ragas", {})
        
        print(f"   📊 Live Scores [{case_id}] \n"
            f"     🔹 Faithfulness: {ragas_scores.get('faithfulness', 0.0):.2f}\n"
            f"     🔹 Relevance: {ragas_scores.get('answer_relevance', 0.0):.2f}\n"
            f"     🔹 Context Recall: {ragas_scores.get('context_recall', 0.0):.2f}\n"
            f"     🔹 Answer Correctness: {ragas_scores.get('answer_correctness', 0.0):.2f}\n")
        
    # --- 💾 NEW COMPONENT: PERSIST DATA TO DISK ---
    output_file = "data/evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_matrix, f, indent=2)
    # ----------------------------------------------
    # 4. Save the compiled results matrix permanently to disk
    output_file = "data/evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_matrix, f, indent=2)

    print("---")
    print("✨ Comprehensive evaluation loop concluded successfully.")
    print(f"💾 Full cross-framework matrix saved to: [{output_file}]")
    print("📊 Verification Checkpoint Matrix Output Sample (Row 1 Summary):")
    print(json.dumps(evaluation_matrix[0]["frameworks"]["ragas"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
