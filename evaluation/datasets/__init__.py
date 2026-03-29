"""Evaluation datasets."""

from .base import BaseEvalDataset
from .flickr30k import Flickr30KRetrievalDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .imagenet import (
    ImageNetDataset,
    ImageNetV2Dataset,
    ImageNetReaLDataset,
    ObjectNetDataset,
)
from .coco import COCORetrievalDataset
from .patternnet import PatternNetDataset

__all__ = [
    "BaseEvalDataset",
    "Flickr30KRetrievalDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "ImageNetDataset",
    "ImageNetV2Dataset",
    "ImageNetReaLDataset",
    "ObjectNetDataset",
    "COCORetrievalDataset",
    "PatternNetDataset",
]
