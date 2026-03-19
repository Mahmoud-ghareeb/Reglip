"""Base class for evaluation tasks."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTask(ABC):
    """Abstract base class for evaluation tasks."""
    
    name: str = "base"
    
    def __init__(self, evaluator):
        """
        Initialize the task.
        
        Args:
            evaluator: Evaluator instance with model and utilities
        """
        self.evaluator = evaluator
    
    @abstractmethod
    def run(self, dataset, **kwargs) -> Dict[str, float]:
        """
        Run the evaluation task.
        
        Args:
            dataset: Dataset to evaluate on
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric names to values
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
