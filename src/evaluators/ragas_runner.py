import os
import requests
import json
from typing import Dict, Any, List
from src.evaluators.base import BaseEvaluator

# Maintain strict local environment routing parameters to block cloud leaks
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

from langchain_ollama import ChatOllama
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall
from ragas.run_config import RunConfig 

# --- 🩹 CLEAN COMPOSITION MIDDLEWARE FOR LOCAL JSON ---
class LocalJsonChatModelMiddleware:
    """
    Wraps the native ChatOllama instance directly. Intercepts calls to invoke/generate
    and dynamically injects the missing 'text' field to satisfy Ragas' internal 
    Pydantic StringIO schemas.
    """
    def __init__(self, raw_llm: ChatOllama):
        self.raw_llm = raw_llm

    def __getattr__(self, name):
        # Proxy all standard method calls right through to the underlying LangChain model
        return getattr(self.raw_llm, name)

    def generate_prompt(self, prompts: List[Any], stop: Any = None, callbacks: Any = None, **kwargs: Any) -> Any:
        # Execute the raw prediction line
        response = self.raw_llm.generate_prompt(prompts, stop, callbacks, **kwargs)
        
        try:
            # Safely navigate down into the generated text payload string
            text_content = response.generations[0][0].text.strip()
            
            # FIX: Intercept classification responses that are missing the parent 'text' token
            if "classifications" in text_content and '"text":' not in text_content:
                parsed_json = json.loads(text_content)
                if "classifications" in parsed_json:
                    # Duplicate the raw data string back inside a safe fallback key structure
                    parsed_json["text"] = text_content
                    response.generations[0][0].text = json.dumps(parsed_json)
        except Exception:
            pass # Fall back to the raw generated string safely if it isn't JSON text
            
        return response
        
    def __call__(self, *args, **kwargs):
        return self.raw_llm(*args, **kwargs)

# --- 🛠️ CUSTOM LOCAL EMBEDDING ROUTER ---
class DirectOllamaEmbeddings:
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = f"{base_url}/api/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.endpoint,
                    json={"model": self.model_name, "prompt": text},
                    timeout=15
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except Exception as e:
                print(f"⚠️ [Embedding Error] Failed vector generation: {str(e)}")
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
# ------------------------------------------------

class RagasRunner(BaseEvaluator):
    def __init__(self, model_name: str = "mistral"):
        self._model_name = model_name
        
        # 1. Main Text Generation Judge (Mistral) configured with forced JSON mode format
        native_ollama = ChatOllama(
            model=self._model_name, 
            base_url="http://localhost:11434", 
            temperature=0.0,
            format="json" 
        )
        
        # Apply the middleware layer onto the native model first, then wrap it for Ragas
        self.patched_llm = LocalJsonChatModelMiddleware(native_ollama)
        self.wrapper_llm = LangchainLLMWrapper(self.patched_llm)
        
        # 2. Custom Embedding Engine (Forced Nomic)
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
                reference=sample["reference"]
            )
            eval_dataset = EvaluationDataset(samples=[ragas_sample])
            
            faith = Faithfulness(llm=self.wrapper_llm)
            relevance = AnswerRelevancy(llm=self.wrapper_llm, embeddings=self.wrapper_embeddings)
            recall = LLMContextRecall(llm=self.wrapper_llm)
            
            local_throttle_config = RunConfig(
                max_workers=1,     
                max_retries=2,     
                max_wait=10        
            )
            
            results = evaluate(
                dataset=eval_dataset,
                metrics=[faith, relevance, recall],
                run_config=local_throttle_config,
                raise_exceptions=True  
            )
            
            scores_df = results.to_pandas()
            
            def safe_float(val):
                import math
                try:
                    float_val = float(val)
                    return 0.0 if math.isnan(float_val) else float_val
                except:
                    return 0.0

            faith_col = "faithfulness" if "faithfulness" in scores_df.columns else scores_df.columns[0]
            relev_col = "answer_relevancy" if "answer_relevancy" in scores_df.columns else "answer_relevance"
            
            recall_col = "context_recall"
            if "llm_context_recall" in scores_df.columns:
                recall_col = "llm_context_recall"

            return {
                "faithfulness": safe_float(scores_df[faith_col].iloc[0]),
                "answer_relevance": safe_float(scores_df[relev_col].iloc[0]),
                "context_recall": safe_float(scores_df[recall_col].iloc[0])
            }
        except Exception as e:
            print(f"❌ [RAGAS RUNTIME EXCEPTION] Row evaluation broken on ID {sample.get('id')}: {str(e)}")
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_recall": 0.0}