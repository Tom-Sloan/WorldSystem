# SLAM3R Tests

This directory contains tests for SLAM3R functionality.

## Test Files

- `test_segment_detection.py` - Tests video segment boundary detection
- `test_segment_reset.py` - Tests SLAM state reset between video segments

## Running Tests

### Prerequisites

1. Activate conda environment:
   ```bash
   conda activate slam3r
   ```

2. Ensure SLAM3R dependencies are installed:
   ```bash
   cd ../SLAM3R_engine
   pip install -r requirements.txt
   ```

### Run Tests

```bash
# Test segment detection
python test_segment_detection.py

# Test segment reset functionality
python test_segment_reset.py
```

## Test Data

These tests may require video segment data. Check the individual test files for specific data requirements.