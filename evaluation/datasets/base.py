"""Base class for evaluation datasets."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset


class BaseEvalDataset(Dataset, ABC):
    """Abstract base class for evaluation datasets."""
    
    name: str = "base"
    task_type: str = "unknown"  # "retrieval" or "classification"
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        raise NotImplementedError
    
    def get_class_names(self) -> Optional[List[str]]:
        """
        Get class names for classification datasets.
        
        Returns:
            List of class names or None if not applicable
        """
        return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, task_type={self.task_type}, len={len(self)})"
