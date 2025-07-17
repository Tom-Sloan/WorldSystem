import cv2
import numpy as np
from typing import Tuple, Dict


class FrameQualityScorer:
    """Scores frame quality based on multiple factors."""
    
    def __init__(self):
        self.weights = {
            'sharpness': 0.35,
            'exposure': 0.25,
            'size': 0.25,
            'centering': 0.15
        }
    
    def calculate_sharpness(self, roi: np.ndarray) -> float:
        """Calculate sharpness score using Laplacian variance."""
        if roi.size == 0:
            return 0.0
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (empirically determined)
        normalized = min(variance / 1000.0, 1.0)
        return normalized
    
    def calculate_exposure(self, roi: np.ndarray) -> float:
        """Calculate exposure quality score."""
        if roi.size == 0:
            return 0.0
            
        # Convert to grayscale if needed
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check for over/under exposure
        underexposed = hist[:20].sum()
        overexposed = hist[235:].sum()
        
        # Calculate entropy as measure of information content
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        
        # Combine metrics
        exposure_score = entropy / 8.0  # Normalize entropy
        exposure_score *= (1.0 - underexposed) * (1.0 - overexposed)
        
        return min(exposure_score, 1.0)
    
    def calculate_size_score(self, bbox: Tuple[int, int, int, int], 
                           frame_shape: Tuple[int, int]) -> float:
        """Calculate score based on object size relative to frame."""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Ideal size is 5-25% of frame
        ratio = bbox_area / frame_area
        if ratio < 0.05:
            score = ratio / 0.05
        elif ratio > 0.25:
            score = max(0, 1.0 - (ratio - 0.25) / 0.25)
        else:
            score = 1.0
            
        return score
    
    def calculate_centering_score(self, bbox: Tuple[int, int, int, int],
                                frame_shape: Tuple[int, int]) -> float:
        """Calculate how well-centered the object is."""
        x1, y1, x2, y2 = bbox
        
        # Calculate center of bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Calculate frame center
        frame_center_x = frame_shape[1] / 2
        frame_center_y = frame_shape[0] / 2
        
        # Calculate normalized distance from center
        dx = abs(bbox_center_x - frame_center_x) / frame_center_x
        dy = abs(bbox_center_y - frame_center_y) / frame_center_y
        
        # Convert to score (closer to center = higher score)
        distance = np.sqrt(dx**2 + dy**2)
        score = max(0, 1.0 - distance)
        
        return score
    
    def score_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   frame_shape: Tuple[int, int]) -> Tuple[float, Dict[str, float]]:
        """Calculate overall quality score for a frame."""
        x1, y1, x2, y2 = bbox
        
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
        
        return total_score, scores