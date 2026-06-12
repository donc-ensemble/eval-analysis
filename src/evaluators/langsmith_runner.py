import json
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, Tuple
from langsmith import Client

OLLAMA_HOST = "http://localhost:11434"

ZERO_SCORES: Dict[str, float] = {
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "context_recall": 0.0,
    "answer_correctness": 0.0,
}


class LangSmithRunner:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.client = Client()

        self.dataset_name = "Langsmith Dataset"
        try:
            self.dataset = self.client.read_dataset(dataset_name=self.dataset_name)
        except Exception:
            self.dataset = self.client.create_dataset(
                dataset_name=self.dataset_name,
                description="Target dataset for multi-framework local RAG evaluations.",
            )

        self.example_map = {}
        try:
            examples = list(self.client.list_examples(dataset_id=self.dataset.id))
            for ex in examples:
                c_id = ex.metadata.get("id") or ex.inputs.get("id")
                if c_id:
                    self.example_map[str(c_id)] = ex.id
        except Exception:
            pass

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_name = f"experiment-{self.model_name}-{ts}"
        try:
            self.project = self.client.create_project(
                project_name=self.project_name,
                reference_dataset_id=self.dataset.id,
                metadata={"model": self.model_name, "timestamp": ts},
            )
        except Exception:
            self.project_name = "default"

    async def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        sample_id = str(sample.get("id", uuid.uuid4().hex))
        user_input = sample.get("user_input", "")
        response = sample.get("response", "")
        reference = sample.get("reference", "")
        contexts = sample.get("retrieved_contexts", [])
        context_str = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)

        if sample_id not in self.example_map:
            try:
                new_ex = self.client.create_example(
                    inputs={"user_input": user_input, "id": sample_id},
                    outputs={"reference": reference},
                    metadata={"id": sample_id},
                    dataset_id=self.dataset.id,
                )
                self.example_map[sample_id] = new_ex.id
            except Exception:
                self.example_map[sample_id] = None

        example_id = self.example_map.get(sample_id)

        run_id = uuid.uuid4()
        try:
            self.client.create_run(
                id=run_id,
                name=f"Eval Row: {sample_id}",
                run_type="chain",
                inputs={
                    "user_input": user_input,
                    "response": response,
                    "reference": reference,
                    "contexts": contexts,
                },
                project_name=self.project_name,
                reference_example_id=example_id,
                start_time=datetime.utcnow(),
            )
        except Exception:
            pass

        prompts = {
            "faithfulness": f"Context: {context_str}\nResponse: {response}\n\nIs the response grounded and faithful to the context without hallucinations?",
            "answer_relevance": f"Question: {user_input}\nResponse: {response}\n\nIs the response directly focused and relevant to the user query?",
            "context_recall": f"Ground Truth Reference: {reference}\nRetrieved Context: {context_str}\n\nDoes the context contain all crucial details needed to form the reference answer?",
            "answer_correctness": f"Response: {response}\nGround Truth Reference: {reference}\n\nRate the semantic accuracy and factual alignment of the response to the ground truth.",
        }

        tasks = [self._get_judge_score(metric, prompt) for metric, prompt in prompts.items()]
        results = await asyncio.gather(*tasks)

        scores = dict(ZERO_SCORES)
        for metric, score in results:
            scores[metric] = score

        try:
            self.client.update_run(
                run_id,
                outputs={"scores": scores},
                end_time=datetime.utcnow(),
            )
            for metric, score in scores.items():
                self.client.create_feedback(run_id=run_id, key=metric, score=score)
        except Exception:
            pass

        return scores

    async def _get_judge_score(self, metric: str, prompt: str) -> Tuple[str, float]:
        import httpx

        judge_prompt = (
            prompt
            + '\n\nAnalyze the data and return ONLY a valid JSON object containing a single key '
            '"score" mapped to a float value from 0.0 to 1.0. Example format: {"score": 0.85}'
        )
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": judge_prompt,
                        "stream": False,
                        "format": "json",
                    },
                    timeout=90.0,
                )
                if response.status_code == 200:
                    res_data = response.json()
                    parsed = json.loads(res_data.get("response", "{}"))
                    score = float(parsed.get("score", 0.0))
                    return metric, min(max(score, 0.0), 1.0)
                else:
                    print(f"⚠️ [LangSmith] Ollama returned {response.status_code} for {metric}")
        except Exception as e:
            print(f"⚠️ [LangSmith] Judge call failed for {metric}: {e}")

        return metric, 0.0
