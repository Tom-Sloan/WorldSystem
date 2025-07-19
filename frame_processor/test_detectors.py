#!/usr/bin/env python3
"""
Test script to compare SAM and FastSAM detectors.
"""

import asyncio
import cv2
import time
import numpy as np
from pathlib import Path

from core.config import Config
from detection.sam import SAMDetector
from detection.fastsam import FastSAMDetector
from detection.yolo import YOLODetector


async def test_detector(detector, image_path: str, name: str):
    """Test a detector on an image."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Warmup
    print("Warming up...")
    detector.warmup()
    
    # Time detection
    print("Running detection...")
    start = time.time()
    detections = await detector.detect(image)
    inference_time = (time.time() - start) * 1000
    
    print(f"\nResults:")
    print(f"- Inference time: {inference_time:.1f}ms")
    print(f"- FPS: {1000/inference_time:.1f}")
    print(f"- Detections: {len(detections)}")
    
    # Analyze detections
    if detections:
        confidences = [d.confidence for d in detections]
        areas = [(d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) for d in detections]
        
        print(f"\nDetection statistics:")
        print(f"- Avg confidence: {np.mean(confidences):.3f}")
        print(f"- Min confidence: {np.min(confidences):.3f}")
        print(f"- Max confidence: {np.max(confidences):.3f}")
        print(f"- Avg area: {np.mean(areas):.0f} pixels")
        print(f"- Total coverage: {sum(areas)/(image.shape[0]*image.shape[1])*100:.1f}%")
    
    # Draw and save results
    result_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"{det.confidence:.2f}", 
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    output_path = f"test_results_{name.lower().replace(' ', '_')}.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nSaved visualization to: {output_path}")
    
    return detections


async def compare_detectors():
    """Compare all detectors on the same image."""
    # Test image - replace with your test image
    test_image = "test_office.jpg"
    
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        print("Creating a test image with various objects...")
        # Create synthetic test image
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        # Add some objects
        cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), -1)  # Black square
        cv2.circle(img, (500, 200), 80, (128, 128, 128), -1)  # Gray circle
        cv2.rectangle(img, (700, 150), (900, 250), (200, 200, 200), -1)  # Light rect
        cv2.imwrite(test_image, img)
    
    # Initialize detectors
    config = Config()
    
    # YOLO baseline
    yolo = YOLODetector(
        model_path="models/yolov11l.pt",
        confidence=0.25,  # Lower threshold
        device="cuda"
    )
    
    # SAM2 Hiera Large
    sam = SAMDetector(
        model_cfg="sam2_hiera_l.yaml",
        model_path="models/sam2_hiera_large.pt",
        device="cuda",
        points_per_side=24
    )
    
    # FastSAM
    fastsam = FastSAMDetector(
        model_path="models/FastSAM-x.pt",
        device="cuda",
        conf_threshold=0.4
    )
    
    # Test each detector
    yolo_results = await test_detector(yolo, test_image, "YOLO v11")
    sam_results = await test_detector(sam, test_image, "SAM2-Hiera-L")
    fastsam_results = await test_detector(fastsam, test_image, "FastSAM")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Detector':<15} {'Detections':<15} {'Unique Objects':<15}")
    print(f"{'-'*45}")
    print(f"{'YOLO v11':<15} {len(yolo_results):<15} {len(set(d.class_name for d in yolo_results)):<15}")
    print(f"{'SAM2-Hiera-L':<15} {len(sam_results):<15} {'All objects':<15}")
    print(f"{'FastSAM':<15} {len(fastsam_results):<15} {'All objects':<15}")


if __name__ == "__main__":
    asyncio.run(compare_detectors())