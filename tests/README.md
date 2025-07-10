# WorldSystem Tests

This directory contains tests for the WorldSystem project components.

## Directory Structure

- `integration/` - Integration tests that test multiple components working together
  - `test_full_integration.py` - Tests full SLAM3R → Mesh Service → Visualization pipeline
  - `test_slam3r_mesh_integration.py` - Tests SLAM3R and Mesh Service integration

## Running Tests

### Prerequisites
1. Ensure conda environment is activated:
   ```bash
   conda activate 3dreconstruction
   ```

2. Ensure required services are running:
   ```bash
   docker-compose --profile mesh_service up -d
   ```

### Integration Tests

To run integration tests:
```bash
cd tests/integration
python test_full_integration.py
python test_slam3r_mesh_integration.py
```

## Test Video

Some tests require a test video file. The tests will look for `test_video.mp4` in:
1. Current directory
2. Project root directory
3. Absolute path: `/home/sam3/Desktop/Toms_Workspace/WorldSystem/test_video.mp4`

You can also provide a video path as an argument to most test scripts.