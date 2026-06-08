from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a sample dictionary containing:
        user_input, retrieved_contexts, response, reference
        
        Returns a standardized dictionary of metrics between 0.0 and 1.0:
        {
            "faithfulness": float,
            "answer_relevance": float,
            "context_recall": float
        }
        """
        pass