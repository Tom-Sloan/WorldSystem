"""
Frame processing pipeline module.

This module contains supporting components for video processing
including scoring, enhancement, and publishing.
"""

from .scorer import FrameScorer
from .enhancer import ImageEnhancer
from .publisher import RabbitMQPublisher

__all__ = [
    'FrameScorer',
    'ImageEnhancer',
    'RabbitMQPublisher'
]