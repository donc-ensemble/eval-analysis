import math
import os
import requests
import asyncio
from typing import Dict, Any, List, Optional
from src.evaluators.base import BaseEvaluator

os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

from langchain_ollama import ChatOllama
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, AnswerCorrectness
from ragas.run_config import RunConfig


def _safe_float(val: Any) -> float:
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _resolve_col(df, *candidates: str) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    print(f"⚠️ [RAGAS] None of {candidates} found in columns: {list(df.columns)}")
    return None


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
                print(f"⚠️ [Embedding Error] Failed vector generation: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


class RagasRunner(BaseEvaluator):
    def __init__(self, model_name: str = "mistral"):
        self._model_name = model_name

        self.native_ollama = ChatOllama(
            model=self._model_name,
            base_url="http://localhost:11434",
            temperature=0.0,
        )
        self.wrapper_llm = LangchainLLMWrapper(self.native_ollama)
        self.local_embeddings = DirectOllamaEmbeddings(model_name="nomic-embed-text")
        self.wrapper_embeddings = LangchainEmbeddingsWrapper(self.local_embeddings)

    @property
    def name(self) -> str:
        return "ragas"

    async def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        zero = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
        }
        try:
            ragas_sample = SingleTurnSample(
                user_input=sample["user_input"],
                retrieved_contexts=sample["retrieved_contexts"],
                response=sample["response"],
                reference=sample["reference"],
            )
            eval_dataset = EvaluationDataset(samples=[ragas_sample])

            results = evaluate(
                dataset=eval_dataset,
                metrics=[Faithfulness(), AnswerRelevancy(), LLMContextRecall(), AnswerCorrectness()],
                llm=self.wrapper_llm,
                embeddings=self.wrapper_embeddings,
                run_config=RunConfig(max_workers=1, max_retries=1, max_wait=5),
                raise_exceptions=False,
            )

            df = results.to_pandas()

            faith_col = _resolve_col(df, "faithfulness")
            relev_col = _resolve_col(df, "answer_relevancy", "answer_relevance")
            recall_col = _resolve_col(df, "context_recall", "llm_context_recall")
            correct_col = _resolve_col(df, "answer_correctness", "answer_similarity")

            return {
                "faithfulness": _safe_float(df[faith_col].iloc[0]) if faith_col else 0.0,
                "answer_relevance": _safe_float(df[relev_col].iloc[0]) if relev_col else 0.0,
                "context_recall": _safe_float(df[recall_col].iloc[0]) if recall_col else 0.0,
                "answer_correctness": _safe_float(df[correct_col].iloc[0]) if correct_col else 0.0,
            }
        except Exception as e:
            print(f"❌ [RAGAS] Exception on {sample.get('id')}: {e}", flush=True)
            return zero
