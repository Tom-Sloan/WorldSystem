"""
Automatic prompting strategies for SAM2 video tracking.

This module provides different strategies for generating prompts to discover
objects in video frames without manual annotation.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import cv2
from dataclasses import dataclass

from core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Prompt:
    """A single prompt point or box."""
    x: float
    y: float
    label: int = 1  # 1 for positive, 0 for negative
    prompt_type: str = "point"  # "point" or "box"
    confidence: float = 1.0


class PromptStrategy(ABC):
    """Base class for automatic prompt generation strategies."""
    
    @abstractmethod
    async def generate_prompts(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Generate prompts for object discovery.
        
        Args:
            frame: Input frame (H, W, C)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary with:
                - points: numpy array of point coordinates (N, 2)
                - labels: numpy array of point labels (N,)
                - strategy: name of the strategy used
                - metadata: optional strategy-specific metadata
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """Validate input frame."""
        if frame is None or frame.size == 0:
            return False
        if len(frame.shape) != 3:
            return False
        return True


class GridPromptStrategy(PromptStrategy):
    """Generate uniform grid of point prompts."""
    
    def __init__(self, points_per_side: int = 16, margin: int = 50):
        self.points_per_side = points_per_side
        self.margin = margin
        logger.info(f"Initialized GridPromptStrategy with {points_per_side} points per side")
        
    async def generate_prompts(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate grid prompts optimized for real-time performance."""
        if not self.validate_frame(frame):
            logger.error("Invalid frame for grid prompting")
            return {"points": np.array([]), "labels": np.array([]), "strategy": self.name}
        
        h, w = frame.shape[:2]
        
        # Adaptive density based on resolution
        density = kwargs.get('density_override', self.points_per_side)
        if h > 1080 or w > 1920:
            density = max(8, density // 2)
            logger.debug(f"Reduced grid density to {density} for high resolution")
        
        # Ensure we have valid margins
        margin = min(self.margin, min(h, w) // 4)
        
        # Generate evenly spaced points
        y_coords = np.linspace(margin, h - margin, density)
        x_coords = np.linspace(margin, w - margin, density)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([int(x), int(y)])
        
        points_array = np.array(points)
        labels_array = np.ones(len(points), dtype=np.int32)
        
        logger.debug(f"Generated {len(points)} grid prompts for {w}x{h} frame")
        
        return {
            "points": points_array,
            "labels": labels_array,
            "strategy": self.name,
            "metadata": {
                "grid_size": density,
                "frame_resolution": (w, h),
                "margin": margin
            }
        }
    
    @property
    def name(self) -> str:
        return "grid"


class MotionPromptStrategy(PromptStrategy):
    """Detect moving regions between frames for prompting."""
    
    def __init__(self, threshold: float = 25.0, min_area: int = 100, 
                 max_prompts: int = 50):
        self.threshold = threshold
        self.min_area = min_area
        self.max_prompts = max_prompts
        self.prev_frame = None
        self.prev_gray = None
        logger.info(f"Initialized MotionPromptStrategy with threshold={threshold}")
        
    async def generate_prompts(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate prompts based on motion detection."""
        if not self.validate_frame(frame):
            logger.error("Invalid frame for motion prompting")
            return {"points": np.array([]), "labels": np.array([]), "strategy": self.name}
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame - use sparse grid
        if self.prev_gray is None:
            self.prev_frame = frame.copy()
            self.prev_gray = gray
            # Return sparse grid for first frame
            grid_strategy = GridPromptStrategy(points_per_side=8)
            result = await grid_strategy.generate_prompts(frame)
            result["strategy"] = f"{self.name}_initial"
            return result
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        
        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Threshold to create binary mask
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate points at motion centers
        points = []
        motion_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])
                motion_areas.append(area)
        
        # Sort by area and limit number of prompts
        if len(points) > self.max_prompts:
            # Keep points with largest motion areas
            sorted_indices = np.argsort(motion_areas)[::-1][:self.max_prompts]
            points = [points[i] for i in sorted_indices]
        
        # Add some grid points if too few motion points
        min_points = kwargs.get('min_points', 10)
        if len(points) < min_points:
            grid_strategy = GridPromptStrategy(points_per_side=6)
            grid_result = await grid_strategy.generate_prompts(frame)
            additional_needed = min_points - len(points)
            if len(grid_result["points"]) > 0:
                # Randomly sample additional points
                indices = np.random.choice(
                    len(grid_result["points"]), 
                    min(additional_needed, len(grid_result["points"])), 
                    replace=False
                )
                for idx in indices:
                    points.append(grid_result["points"][idx].tolist())
        
        # Update previous frame
        self.prev_frame = frame.copy()
        self.prev_gray = gray
        
        points_array = np.array(points) if points else np.array([]).reshape(0, 2)
        labels_array = np.ones(len(points), dtype=np.int32) if points else np.array([])
        
        logger.debug(f"Generated {len(points)} motion prompts (detected {len(contours)} motion regions)")
        
        return {
            "points": points_array,
            "labels": labels_array,
            "strategy": self.name,
            "metadata": {
                "motion_regions": len(contours),
                "total_motion_area": sum(motion_areas) if motion_areas else 0,
                "threshold": self.threshold
            }
        }
    
    def reset(self):
        """Reset motion detection state."""
        self.prev_frame = None
        self.prev_gray = None
        logger.debug("Reset motion detection state")
    
    @property
    def name(self) -> str:
        return "motion"


class SaliencyPromptStrategy(PromptStrategy):
    """Generate prompts based on visual saliency."""
    
    def __init__(self, num_prompts: int = 20):
        self.num_prompts = num_prompts
        logger.info(f"Initialized SaliencyPromptStrategy with {num_prompts} prompts")
        
    async def generate_prompts(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate prompts based on saliency detection."""
        if not self.validate_frame(frame):
            logger.error("Invalid frame for saliency prompting")
            return {"points": np.array([]), "labels": np.array([]), "strategy": self.name}
        
        h, w = frame.shape[:2]
        
        # Create saliency detector
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        
        # Compute saliency map
        success, saliency_map = saliency.computeSaliency(frame)
        
        if not success or saliency_map is None:
            logger.warning("Saliency computation failed, falling back to grid")
            grid_strategy = GridPromptStrategy(points_per_side=8)
            result = await grid_strategy.generate_prompts(frame)
            result["strategy"] = f"{self.name}_fallback"
            return result
        
        # Normalize saliency map
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        # Apply threshold to get salient regions
        _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours of salient regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        saliency_scores = []
        
        for contour in contours:
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Sample points within bounding box
            num_samples = max(1, min(5, w_box * h_box // 1000))
            for _ in range(num_samples):
                px = np.random.randint(x, x + w_box)
                py = np.random.randint(y, y + h_box)
                
                # Check if point is inside contour
                if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                    points.append([px, py])
                    saliency_scores.append(saliency_map[py, px])
        
        # Sort by saliency score and limit
        if len(points) > self.num_prompts:
            sorted_indices = np.argsort(saliency_scores)[::-1][:self.num_prompts]
            points = [points[i] for i in sorted_indices]
        
        # Add some random points if too few salient points
        if len(points) < self.num_prompts // 2:
            num_random = self.num_prompts // 2 - len(points)
            for _ in range(num_random):
                px = np.random.randint(50, w - 50)
                py = np.random.randint(50, h - 50)
                points.append([px, py])
        
        points_array = np.array(points) if points else np.array([]).reshape(0, 2)
        labels_array = np.ones(len(points), dtype=np.int32) if points else np.array([])
        
        logger.debug(f"Generated {len(points)} saliency prompts")
        
        return {
            "points": points_array,
            "labels": labels_array,
            "strategy": self.name,
            "metadata": {
                "salient_regions": len(contours),
                "max_saliency": float(np.max(saliency_map)) if saliency_map.size > 0 else 0
            }
        }
    
    @property
    def name(self) -> str:
        return "saliency"


class HybridPromptStrategy(PromptStrategy):
    """Combine multiple prompting strategies."""
    
    def __init__(self, strategies: List[PromptStrategy], weights: Optional[List[float]] = None):
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        strategy_names = [s.name for s in strategies]
        logger.info(f"Initialized HybridPromptStrategy with strategies: {strategy_names}")
        
    async def generate_prompts(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Combine prompts from multiple strategies."""
        if not self.validate_frame(frame):
            logger.error("Invalid frame for hybrid prompting")
            return {"points": np.array([]), "labels": np.array([]), "strategy": self.name}
        
        all_points = []
        all_labels = []
        metadata = {}
        
        # Collect prompts from each strategy
        for i, (strategy, weight) in enumerate(zip(self.strategies, self.weights)):
            try:
                result = await strategy.generate_prompts(frame, **kwargs)
                
                if len(result["points"]) > 0:
                    # Sample points based on weight
                    n_points = int(len(result["points"]) * weight)
                    if n_points > 0:
                        # Randomly sample points
                        indices = np.random.choice(
                            len(result["points"]), 
                            min(n_points, len(result["points"])), 
                            replace=False
                        )
                        
                        selected_points = result["points"][indices]
                        selected_labels = result["labels"][indices]
                        
                        all_points.extend(selected_points.tolist())
                        all_labels.extend(selected_labels.tolist())
                        
                        # Store metadata
                        metadata[strategy.name] = {
                            "selected": n_points,
                            "total": len(result["points"]),
                            "metadata": result.get("metadata", {})
                        }
                        
            except Exception as e:
                logger.error(f"Error in {strategy.name} strategy: {e}")
                continue
        
        # Remove duplicate points (within threshold)
        if all_points:
            points_array = np.array(all_points)
            labels_array = np.array(all_labels)
            
            # Remove duplicates
            unique_points, unique_indices = self._remove_duplicates(points_array, threshold=10)
            unique_labels = labels_array[unique_indices]
            
            logger.debug(f"Generated {len(unique_points)} hybrid prompts "
                        f"(reduced from {len(all_points)})")
            
            return {
                "points": unique_points,
                "labels": unique_labels,
                "strategy": self.name,
                "metadata": metadata
            }
        
        return {
            "points": np.array([]).reshape(0, 2),
            "labels": np.array([]),
            "strategy": self.name,
            "metadata": metadata
        }
    
    def _remove_duplicates(self, points: np.ndarray, threshold: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicate points within threshold distance."""
        if len(points) == 0:
            return points, np.array([])
        
        unique_indices = []
        unique_points = []
        
        for i, point in enumerate(points):
            is_duplicate = False
            for unique_point in unique_points:
                dist = np.linalg.norm(point - unique_point)
                if dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_indices.append(i)
                unique_points.append(point)
        
        return np.array(unique_points), np.array(unique_indices)
    
    @property
    def name(self) -> str:
        return f"hybrid({'+'.join(s.name for s in self.strategies)})"


# Factory function for creating prompt strategies
def create_prompt_strategy(strategy_name: str, **kwargs) -> PromptStrategy:
    """
    Factory function to create prompt strategies.
    
    Args:
        strategy_name: Name of the strategy ("grid", "motion", "saliency", "hybrid")
        **kwargs: Strategy-specific parameters
        
    Returns:
        PromptStrategy instance
    """
    strategies = {
        "grid": lambda: GridPromptStrategy(
            points_per_side=kwargs.get("points_per_side", 16),
            margin=kwargs.get("margin", 50)
        ),
        "motion": lambda: MotionPromptStrategy(
            threshold=kwargs.get("threshold", 25.0),
            min_area=kwargs.get("min_area", 100)
        ),
        "saliency": lambda: SaliencyPromptStrategy(
            num_prompts=kwargs.get("num_prompts", 20)
        ),
        "hybrid": lambda: HybridPromptStrategy(
            strategies=kwargs.get("strategies", [
                GridPromptStrategy(),
                MotionPromptStrategy()
            ]),
            weights=kwargs.get("weights", None)
        )
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Unknown strategy {strategy_name}, defaulting to grid")
        strategy_name = "grid"
    
    return strategies[strategy_name]()