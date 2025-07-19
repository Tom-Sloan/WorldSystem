"""
Frame quality scoring module.

This module evaluates frame quality based on sharpness, exposure, size, and centering
to help select the best frame for API processing.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional

from core.utils import get_logger


logger = get_logger(__name__)


class FrameScorer:
    """
    Scores frame quality based on multiple factors.
    
    This preserves the exact scoring logic from the original implementation
    to ensure consistent best frame selection.
    """
    
    def __init__(self, 
                 sharpness_weight: float = 0.35,
                 exposure_weight: float = 0.25,
                 size_weight: float = 0.25,
                 centering_weight: float = 0.15):
        """
        Initialize frame scorer with configurable weights.
        
        Args:
            sharpness_weight: Weight for sharpness score
            exposure_weight: Weight for exposure quality
            size_weight: Weight for object size score
            centering_weight: Weight for centering score
        """
        self.weights = {
            'sharpness': sharpness_weight,
            'exposure': exposure_weight,
            'size': size_weight,
            'centering': centering_weight
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def calculate_sharpness(self, roi: np.ndarray) -> float:
        """
        Calculate sharpness score using Laplacian variance.
        
        Higher variance indicates sharper image.
        
        Args:
            roi: Region of interest to evaluate
            
        Returns:
            Sharpness score between 0 and 1
        """
        if roi.size == 0:
            return 0.0
        
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Calculate Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize to 0-1 range (empirically determined threshold)
            normalized = min(variance / 1000.0, 1.0)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0
    
    def calculate_exposure(self, roi: np.ndarray) -> float:
        """
        Calculate exposure quality score.
        
        Evaluates histogram distribution and entropy.
        
        Args:
            roi: Region of interest to evaluate
            
        Returns:
            Exposure score between 0 and 1
        """
        if roi.size == 0:
            return 0.0
        
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Check for over/under exposure
            underexposed = hist[:20].sum()  # Very dark pixels
            overexposed = hist[235:].sum()   # Very bright pixels
            
            # Calculate entropy as measure of information content
            hist_nonzero = hist[hist > 0]
            if len(hist_nonzero) == 0:
                return 0.0
                
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            
            # Combine metrics
            exposure_score = entropy / 8.0  # Normalize entropy (max ~8 for uniform dist)
            exposure_score *= (1.0 - underexposed) * (1.0 - overexposed)
            
            return min(exposure_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating exposure: {e}")
            return 0.0
    
    def calculate_size_score(self, bbox: Tuple[int, int, int, int], 
                           frame_shape: Tuple[int, int]) -> float:
        """
        Calculate score based on object size relative to frame.
        
        Ideal size is 5-25% of frame area.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Size score between 0 and 1
        """
        try:
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_shape[0] * frame_shape[1]
            
            if frame_area == 0:
                return 0.0
            
            # Calculate ratio
            ratio = bbox_area / frame_area
            
            # Score based on ideal range (5-25% of frame)
            if ratio < 0.05:
                score = ratio / 0.05  # Linear increase up to ideal minimum
            elif ratio > 0.25:
                score = max(0, 1.0 - (ratio - 0.25) / 0.25)  # Linear decrease after ideal maximum
            else:
                score = 1.0  # Perfect size range
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating size score: {e}")
            return 0.0
    
    def calculate_centering_score(self, bbox: Tuple[int, int, int, int],
                                frame_shape: Tuple[int, int]) -> float:
        """
        Calculate how well-centered the object is.
        
        Objects closer to frame center score higher.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Centering score between 0 and 1
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Calculate center of bbox
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            # Calculate frame center
            frame_center_x = frame_shape[1] / 2
            frame_center_y = frame_shape[0] / 2
            
            if frame_center_x == 0 or frame_center_y == 0:
                return 0.0
            
            # Calculate normalized distance from center
            dx = abs(bbox_center_x - frame_center_x) / frame_center_x
            dy = abs(bbox_center_y - frame_center_y) / frame_center_y
            
            # Convert to score (closer to center = higher score)
            distance = np.sqrt(dx**2 + dy**2)
            score = max(0, 1.0 - distance)
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating centering score: {e}")
            return 0.0
    
    def score_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   frame_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall quality score for a frame.
        
        Args:
            frame: Full frame image
            bbox: Bounding box of object (x1, y1, x2, y2)
            frame_shape: Optional frame shape, extracted from frame if not provided
            
        Returns:
            Tuple of (total_score, component_scores_dict)
        """
        if frame_shape is None:
            frame_shape = frame.shape[:2]
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Validate bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Calculate individual scores
            scores = {
                'sharpness': self.calculate_sharpness(roi),
                'exposure': self.calculate_exposure(roi),
                'size': self.calculate_size_score(bbox, frame_shape),
                'centering': self.calculate_centering_score(bbox, frame_shape)
            }
            
            # Calculate weighted total
            total_score = sum(scores[key] * self.weights[key] for key in scores)
            
            logger.debug(
                f"Frame score: {total_score:.3f} "
                f"(sharp:{scores['sharpness']:.2f}, "
                f"expo:{scores['exposure']:.2f}, "
                f"size:{scores['size']:.2f}, "
                f"center:{scores['centering']:.2f})"
            )
            
            return total_score, scores
            
        except Exception as e:
            logger.error(f"Error scoring frame: {e}")
            return 0.0, {
                'sharpness': 0.0,
                'exposure': 0.0,
                'size': 0.0,
                'centering': 0.0
            }