# SAM2 and FastSAM Usage Guide

## Quick Start

### 1. Install Model Weights

First, download the required model weights:

```bash
cd frame_processor
./install_models.sh
```

This will download:
- SAM2 Hiera Large model: `sam2_hiera_large.pt`
- FastSAM-x model (138MB): `FastSAM-x.pt`

SAM2 model sizes available:
- Hiera Tiny (38.9M params)
- Hiera Small (46.0M params)
- Hiera Base+ (80.8M params)
- Hiera Large (224.4M params) - Default

### 2. Switch Between Detectors

#### Using Environment Variables

```bash
# Use SAM2 (best quality, slower)
export DETECTOR_TYPE=sam
docker-compose --profile frame_processor up frame_processor

# Use FastSAM (faster, good quality)
export DETECTOR_TYPE=fastsam
docker-compose --profile frame_processor up frame_processor

# Use YOLO (fastest, limited to 80 classes)
export DETECTOR_TYPE=yolo
docker-compose --profile frame_processor up frame_processor
```

#### Using .env File

Add to your `.env` file:
```env
# Detector selection
DETECTOR_TYPE=sam  # Options: yolo, sam, fastsam

# SAM2 model selection
SAM_MODEL_CFG=sam2_hiera_l.yaml  # Options: sam2_hiera_t.yaml, sam2_hiera_s.yaml, sam2_hiera_b+.yaml, sam2_hiera_l.yaml

# SAM2 tuning
SAM_POINTS_PER_SIDE=32  # Increase for better coverage
SAM_MIN_MASK_REGION_AREA=300  # Decrease to catch smaller objects

# FastSAM tuning
FASTSAM_CONF_THRESHOLD=0.3  # Lower to detect more objects
```

### 3. Test the Detectors

Run the comparison script:
```bash
cd frame_processor
python test_detectors.py
```

This will test all three detectors on a sample image and show:
- Detection count
- Inference time/FPS
- Detection statistics
- Visual results saved as images

## Performance Comparison

| Detector | FPS   | Quality | Use Case |
|----------|-------|---------|----------|
| YOLO v11 | 30-40 | Good for 80 classes | Real-time, known objects |
| FastSAM  | 15-20 | Very good, all objects | Real-time, unknown objects |
| SAM2-L   | 8-12  | Excellent, all objects | Better performance than SAM1 |
| SAM2-B+  | 12-15 | Very good, all objects | Good balance |
| SAM2-S   | 18-22 | Good, all objects | Faster option |
| SAM2-T   | 25-30 | Good, all objects | Fastest SAM2 option |

## Configuration Options

### SAM2 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM_MODEL_CFG` | `sam2_hiera_l.yaml` | Model config: `sam2_hiera_t.yaml`, `sam2_hiera_s.yaml`, `sam2_hiera_b+.yaml`, `sam2_hiera_l.yaml` |
| `SAM_CHECKPOINT_PATH` | `/app/models/sam2_hiera_large.pt` | Path to model weights |
| `SAM_POINTS_PER_SIDE` | 24 | Grid density (higher = more detections) |
| `SAM_PRED_IOU_THRESH` | 0.86 | Quality threshold |
| `SAM_STABILITY_SCORE_THRESH` | 0.92 | Mask stability threshold |
| `SAM_MIN_MASK_REGION_AREA` | 500 | Minimum object size in pixels |

### FastSAM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTSAM_CONF_THRESHOLD` | 0.4 | Detection confidence |
| `FASTSAM_IOU_THRESHOLD` | 0.9 | NMS overlap threshold |
| `FASTSAM_MAX_DET` | 300 | Maximum detections per image |

## Key Differences

### YOLO
- ✅ Fastest (30+ FPS)
- ✅ Returns specific class names (person, chair, etc.)
- ❌ Limited to 80 predefined classes
- ❌ Misses unusual objects

### SAM2
- ✅ Detects ANY object (class-agnostic)
- ✅ Highest quality segmentation
- ✅ Better performance than SAM1 (up to 6x faster)
- ✅ Multiple model sizes for different speed/quality tradeoffs
- ✅ Best for discovering unknown objects
- ❌ Still slower than YOLO
- ❌ Returns generic "object" class

### FastSAM
- ✅ Good balance of speed and quality
- ✅ Detects any object like SAM
- ✅ Faster than SAM2-L but comparable to SAM2-S
- ❌ Slightly lower quality than SAM2
- ❌ Returns generic "object" class

## Troubleshooting

### CUDA Out of Memory
```bash
# Use a smaller SAM2 model
export SAM_MODEL_CFG=sam2_hiera_t.yaml
export SAM_CHECKPOINT_PATH=/app/models/sam2_hiera_tiny.pt

# Or reduce SAM2 grid density
export SAM_POINTS_PER_SIDE=16

# Reduce FastSAM max detections
export FASTSAM_MAX_DET=200
```

### Too Many False Positives
```bash
# Increase SAM quality threshold
export SAM_PRED_IOU_THRESH=0.90

# Increase FastSAM confidence
export FASTSAM_CONF_THRESHOLD=0.5
```

### Missing Small Objects
```bash
# Decrease minimum area
export SAM_MIN_MASK_REGION_AREA=100
```

## Integration with Scene Scaling

When using SAM/FastSAM:
1. All detected objects are labeled as "object"
2. The Google Lens API will attempt to identify each object
3. Perplexity will look up dimensions for identified objects
4. Scene scale is calculated from known object dimensions

This allows the system to discover and measure objects that YOLO would miss, like:
- Thermostats
- Outlets
- Light switches
- Paper/documents
- Custom hardware
- Unusual furniture

## Best Practices

1. **For Maximum Speed**: Use YOLO for known environments
2. **For Discovery**: Use SAM2/FastSAM to find all objects
3. **For Production**: Start with FastSAM or SAM2-T, switch to SAM2-L for critical frames
4. **For Development**: Use SAM2-L to understand what objects are present
5. **For Balance**: SAM2-S offers good middle ground between speed and quality

## Example Usage in Code

```python
from core.config import Config

# Configure for SAM2 Large
config = Config()
config.detector_type = "sam"
config.sam_model_cfg = "sam2_hiera_l.yaml"
config.sam_checkpoint_path = "/app/models/sam2_hiera_large.pt"
config.sam_points_per_side = 32
config.sam_min_mask_region_area = 200

# Configure for SAM2 Tiny (faster)
config.detector_type = "sam"
config.sam_model_cfg = "sam2_hiera_t.yaml"
config.sam_checkpoint_path = "/app/models/sam2_hiera_tiny.pt"

# Configure for FastSAM
config.detector_type = "fastsam"
config.fastsam_conf_threshold = 0.3
config.fastsam_max_det = 500
```