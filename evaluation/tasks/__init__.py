"""Evaluation tasks."""

from .base import BaseTask
from .retrieval import RetrievalTask
from .zero_shot import ZeroShotTask

__all__ = [
    "BaseTask",
    "RetrievalTask",
    "ZeroShotTask",
]
