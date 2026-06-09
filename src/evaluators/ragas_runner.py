import os
import requests
import asyncio
from typing import Dict, Any, List
from src.evaluators.base import BaseEvaluator

os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

from langchain_ollama import ChatOllama
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextRecall,
    ContextPrecision,
    AnswerCorrectness,
)
from ragas.run_config import RunConfig


# --- 🛠️ ASYNC-COMPLIANT LOCAL EMBEDDING ROUTER ---
class DirectOllamaEmbeddings:
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.endpoint = f"{base_url}/api/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.endpoint,
                    json={"model": self.model_name, "prompt": text},
                    timeout=15,
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except Exception as e:
                print(f"⚠️ [Embedding Error] Failed vector generation: {str(e)}")
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


# -----------------------------------------------------------------


class RagasRunner(BaseEvaluator):
    def __init__(self, model_name: str = "mistral"):
        self._model_name = model_name

        # Initialize the clean base ChatOllama model with JSON format enforcement
        self.native_ollama = ChatOllama(
            model=self._model_name,
            base_url="http://localhost:11434",
            temperature=0.0,
            # format="json",
        )
        self.wrapper_llm = LangchainLLMWrapper(self.native_ollama)

        self.local_embeddings = DirectOllamaEmbeddings(model_name="nomic-embed-text")
        self.wrapper_embeddings = LangchainEmbeddingsWrapper(self.local_embeddings)

    @property
    def name(self) -> str:
        return "ragas"

    async def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ragas_sample = SingleTurnSample(
                user_input=sample["user_input"],
                retrieved_contexts=sample["retrieved_contexts"],
                response=sample["response"],
                reference=sample["reference"],
            )
            eval_dataset = EvaluationDataset(samples=[ragas_sample])

            # Instantiate metrics cleanly
            faith = Faithfulness()
            relevance = AnswerRelevancy()
            recall = LLMContextRecall()
            precision = ContextPrecision()
            correctness = AnswerCorrectness()

            local_throttle_config = RunConfig(max_workers=1, max_retries=1, max_wait=5)

            # Pass llm and embeddings globally inside evaluate() to prevent OpenAI context leakage
            results = evaluate(
                dataset=eval_dataset,
                metrics=[faith, relevance, recall, precision, correctness],
                llm=self.wrapper_llm,
                embeddings=self.wrapper_embeddings,
                run_config=local_throttle_config,
                # raise_exceptions=True,
                raise_exceptions=False
            )

            scores_df = results.to_pandas()

            def safe_float(val):
                import math

                try:
                    float_val = float(val)
                    return 0.0 if math.isnan(float_val) else float_val
                except:  # noqa: E722
                    return 0.0

            faith_col = (
                "faithfulness"
                if "faithfulness" in scores_df.columns
                else scores_df.columns[0]
            )
            relev_col = (
                "answer_relevancy"
                if "answer_relevancy" in scores_df.columns
                else "answer_relevance"
            )
            recall_col = (
                "context_recall"
                if "context_recall" in scores_df.columns
                else "llm_context_recall"
            )
            correctness_col = (
                "answer_correctness"
                if "answer_correctness" in scores_df.columns
                else "answer_similarity"
            )

            return { 
                "faithfulness": safe_float(scores_df[faith_col].iloc[0]),
                "answer_relevance": safe_float(scores_df[relev_col].iloc[0]),
                "context_recall": safe_float(scores_df[recall_col].iloc[0]),
                "answer_correctness": safe_float(scores_df[correctness_col].iloc[0]),
            }
        except Exception as e:
            print(
                f"❌ [RAGAS RUNTIME EXCEPTION] Row evaluation broken on ID {sample.get('id')}: {str(e)}"
            )
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
            }
