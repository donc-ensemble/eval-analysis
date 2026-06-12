import asyncio
import json
import os
import subprocess
import tempfile
from typing import Any, Dict
import yaml

from src.evaluators.base import BaseEvaluator

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

ASSERTION_TO_METRIC = {
    "context-faithfulness": "faithfulness",
    "context-recall": "context_recall",
    "answer-relevance": "answer_relevance",
    "factuality": "answer_correctness",
}

ZERO_SCORES: Dict[str, float] = {
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "context_recall": 0.0,
    "answer_correctness": 0.0,
}


class PromptfooRunner(BaseEvaluator):
    def __init__(self, model_name: str = "mistral"):
        self._model_name = model_name
        self._name = "promptfoo"

    @property
    def name(self) -> str:
        return self._name

    def _build_config(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "description": f"RAG eval — {sample.get('id', 'unknown')}",
            "prompts": ["{{response}}"],
            "providers": [{"id": "echo"}],
            "defaultTest": {
                "options": {
                    "provider": {
                        "text": {
                            "id": f"ollama:chat:{self._model_name}",
                            "config": {
                                "baseUrl": OLLAMA_HOST,
                                "temperature": 0.0,
                            },
                        },
                        "embedding": {
                            "id": "ollama:embeddings:nomic-embed-text",
                            "config": {"baseUrl": OLLAMA_HOST},
                        },
                    }
                }
            },
            "tests": [
                {
                    "vars": {
                        "query": sample["user_input"],
                        "context": sample["retrieved_contexts"],
                        "response": sample["response"],
                        "reference": sample["reference"],
                    },
                    "assert": [
                        {"type": "context-faithfulness", "threshold": 0},
                        {"type": "context-recall", "value": "{{reference}}", "threshold": 0},
                        {"type": "answer-relevance", "threshold": 0},
                        {"type": "factuality", "value": "{{reference}}", "threshold": 0},
                    ],
                }
            ],
        }

    def _run_eval(self, config: Dict[str, Any], case_id: str) -> Dict[str, float]:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "promptfooconfig.yaml")
            output_path = os.path.join(tmpdir, "results.json")

            with open(config_path, "w") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            cmd = [
                "npx", "promptfoo", "eval",
                "--config", config_path,
                "--output", output_path,
                "--no-cache",
                "--no-write",
                "--max-concurrency", "1",
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=360)

                # Exit code 100 means assertion failed (not a crash)
                if proc.returncode not in [0, 100]:
                    print(f"⚠️  [Promptfoo] CLI exited {proc.returncode} on [{case_id}]")
                    if proc.stderr:
                        print(f"    stderr: {proc.stderr[:500]}")
                    return dict(ZERO_SCORES)

            except subprocess.TimeoutExpired:
                print(f"⚠️  [Promptfoo] Timeout on [{case_id}]")
                return dict(ZERO_SCORES)
            except Exception as e:
                print(f"⚠️  [Promptfoo] Subprocess error on [{case_id}]: {e}")
                return dict(ZERO_SCORES)

            try:
                with open(output_path) as f:
                    data = json.load(f)
            except Exception:
                return dict(ZERO_SCORES)

        scores = dict(ZERO_SCORES)
        try:
            results_list = data.get("results", {}).get("results", [])
            if not results_list:
                return scores

            component_results = (
                results_list[0].get("gradingResult", {}).get("componentResults", [])
            )

            for cr in component_results:
                assertion_type = cr.get("assertion", {}).get("type", "")
                metric_key = ASSERTION_TO_METRIC.get(assertion_type)
                if metric_key:
                    raw_score = cr.get("score")
                    if raw_score is not None:
                        try:
                            scores[metric_key] = float(raw_score)
                        except (TypeError, ValueError):
                            pass

        except Exception as e:
            print(f"⚠️  [Promptfoo] Score extraction error on [{case_id}]: {e}")

        return scores

    async def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        case_id = sample.get("id", "unknown")
        config = self._build_config(sample)
        return await asyncio.to_thread(self._run_eval, config, case_id)
