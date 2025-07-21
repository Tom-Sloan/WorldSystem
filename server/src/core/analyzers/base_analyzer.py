from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseAnalyzer(ABC):
    """Base class for all video analysis implementations."""
    
    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for analysis."""
        pass
    
    @abstractmethod
    def analyze(self, tensor: torch.Tensor) -> any:
        """Analyze the preprocessed frame."""
        pass
    
    @abstractmethod
    def postprocess(self, results: any, original_frame: np.ndarray) -> np.ndarray:
        """Process results and draw on original frame."""
        pass 