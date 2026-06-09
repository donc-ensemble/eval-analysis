from typing import List, Dict, Any
from src.evaluators.ragas_runner import RagasRunner

class EvaluationOrchestrator:
    def __init__(self, model_name: str = "mistral"):
        print(f"🤖 Initializing evaluation pipelines using local model: [{model_name}]")
        self.ragas_runner = RagasRunner(model_name=model_name)

    async def run_full_suite(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compiled_results = []
        
        for sample in dataset:
            case_id = sample.get("id")
            print(f"⚙️ Processing row [{case_id}]...")
            
            # Run Ragas execution pass
            ragas_scores = await self.ragas_runner.evaluate_sample(sample)
            
            collated_row = {
                "id": case_id,
                "user_input": sample["user_input"],
                "retrieved_contexts": sample["retrieved_contexts"],
                "response": sample["response"],
                "reference": sample["reference"],
                "frameworks": {
                    "ragas": {
                        "answer_correctness": ragas_scores.get("answer_correctness", 0.0),
                        "faithfulness": ragas_scores.get("faithfulness", 0.0),
                        "answer_relevance": ragas_scores.get("answer_relevance", 0.0),
                        "context_recall": ragas_scores.get("context_recall", 0.0)
                    },
                    "promptfoo": {
                        "faithfulness": 0.0,
                        "answer_relevance": 0.0,
                        "context_recall": 0.0
                    },
                    "langsmith": {
                        "faithfulness": 0.0,
                        "answer_relevance": 0.0,
                        "context_recall": 0.0
                    }
                }
            }
            compiled_results.append(collated_row)
            
        print("\n✨ All metrics collected successfully across active runner modules.")
        return compiled_results