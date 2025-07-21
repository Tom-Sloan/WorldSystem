import cv2
import torch
import numpy as np
from .base_analyzer import BaseAnalyzer
from src.config.settings import DEVICE, CUDA_AVAILABLE, DEBUG_MODE, logger

class YOLOAnalyzer(BaseAnalyzer):
    def __init__(self, model):
        self.model = model
        if CUDA_AVAILABLE:
            self.model.to(DEVICE)
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for YOLO."""
        # Resize to 640x640
        frame_resized = cv2.resize(frame, (640, 640))
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Transpose to BCHW format
        frame_transposed = frame_rgb.transpose(2, 0, 1)
        # Add batch dimension and normalize to 0-1
        frame_batch = np.expand_dims(frame_transposed, 0) / 255.0
        
        if CUDA_AVAILABLE:
            return torch.from_numpy(frame_batch).float().to(DEVICE)
        return torch.from_numpy(frame_batch).float()
    
    def analyze(self, tensor: torch.Tensor) -> any:
        """Run YOLO inference."""
        return self.model(tensor, conf=0.5, verbose=DEBUG_MODE)
    
    def postprocess(self, results: any, original_frame: np.ndarray) -> np.ndarray:
        """Draw YOLO results on frame."""
        frame = original_frame.copy()
        orig_h, orig_w = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Scale coordinates back to original frame size
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                x1 = int(x1 * orig_w / 640)
                x2 = int(x2 * orig_w / 640)
                y1 = int(y1 * orig_h / 640)
                y2 = int(y2 * orig_h / 640)
                
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 255, 0), 2)
        
        return frame 