import numpy as np
import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DimensionEstimate:
    """Represents a dimension estimate for scene scaling."""
    object_id: int
    class_name: str
    product_name: str
    width_m: float
    height_m: float
    depth_m: float
    confidence: float
    detection_confidence: float  # YOLO confidence
    combined_confidence: float  # Product of API confidence and detection confidence


class SceneScaler:
    """Calculates scene scale based on identified object dimensions."""
    
    def __init__(self, min_confidence=0.7):
        self.min_confidence = min_confidence
        self.dimension_estimates: List[DimensionEstimate] = []
        self.current_scale_factor = 1.0
        
    def add_dimension_estimate(self, object_id: int, class_name: str,
                             dimension_data: Dict, detection_confidence: float):
        """Add a new dimension estimate from API results."""
        if dimension_data and 'dimensions' in dimension_data:
            dims = dimension_data['dimensions']
            api_confidence = dimension_data.get('confidence', 0.5)
            combined_confidence = api_confidence * detection_confidence
            
            estimate = DimensionEstimate(
                object_id=object_id,
                class_name=class_name,
                product_name=dimension_data.get('product_name', 'Unknown'),
                width_m=dims.get('width_m', 0),
                height_m=dims.get('height_m', 0),
                depth_m=dims.get('depth_m', 0),
                confidence=api_confidence,
                detection_confidence=detection_confidence,
                combined_confidence=combined_confidence
            )
            
            self.dimension_estimates.append(estimate)
            
    def calculate_weighted_scale(self) -> Dict:
        """Calculate weighted average scale factor based on all estimates."""
        # Filter by minimum confidence
        valid_estimates = [e for e in self.dimension_estimates 
                          if e.combined_confidence >= self.min_confidence]
        
        if not valid_estimates:
            return {
                'scale_factor': 1.0,
                'confidence': 0.0,
                'num_estimates': 0,
                'estimates': []
            }
        
        # Calculate weighted averages for each dimension
        total_weight = sum(e.combined_confidence for e in valid_estimates)
        
        avg_width = sum(e.width_m * e.combined_confidence for e in valid_estimates) / total_weight
        avg_height = sum(e.height_m * e.combined_confidence for e in valid_estimates) / total_weight
        avg_depth = sum(e.depth_m * e.combined_confidence for e in valid_estimates) / total_weight
        
        # Average dimension for scale calculation
        avg_dimension = (avg_width + avg_height + avg_depth) / 3
        
        # Assuming the 3D reconstruction units are arbitrary, we set a scale
        # where 1 unit = avg_dimension meters
        scale_factor = avg_dimension
        
        # Calculate confidence of the scale estimate
        avg_confidence = total_weight / len(valid_estimates)
        
        return {
            'scale_factor': scale_factor,
            'avg_dimensions_m': {
                'width': avg_width,
                'height': avg_height,
                'depth': avg_depth
            },
            'confidence': avg_confidence,
            'num_estimates': len(valid_estimates),
            'estimates': [
                {
                    'object_id': e.object_id,
                    'product': e.product_name,
                    'dimensions_m': {
                        'width': e.width_m,
                        'height': e.height_m,
                        'depth': e.depth_m
                    },
                    'confidence': e.combined_confidence
                }
                for e in valid_estimates
            ]
        }
    
    def get_scale_for_mesh_service(self) -> Dict:
        """Get scale information formatted for mesh service."""
        scale_info = self.calculate_weighted_scale()
        
        return {
            'scale_factor': scale_info['scale_factor'],
            'units_per_meter': 1.0 / scale_info['scale_factor'] if scale_info['scale_factor'] > 0 else 1.0,
            'confidence': scale_info['confidence'],
            'source': 'object_dimensions',
            'timestamp': int(time.time() * 1e9)  # nanoseconds
        }
    
    def should_update_scale(self, new_confidence: float) -> bool:
        """Determine if scale should be updated based on new estimate confidence."""
        current_scale_info = self.calculate_weighted_scale()
        return new_confidence > current_scale_info['confidence'] * 1.1  # 10% improvement threshold