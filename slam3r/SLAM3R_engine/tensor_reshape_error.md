# SLAM3R Tensor Reshape Error - Reproduction Guide

## Error Description

**Error Message**: `RuntimeError: shape '[25, 196, 12, 64]' is invalid for input of size 752640`

**Location**: Occurs in the multiview attention blocks during L2W (Local-to-World) inference, specifically when reshaping query tensors in the cross-attention mechanism.

**When It Occurs**: The error happens on frame 6 of real video processing - the first incremental processing frame after the 5-frame bootstrap completes successfully.

## Analysis

The error indicates a dimension mismatch:
- Expected shape: `[25, 196, 12, 64]` = 25 × 196 × 12 × 64 = 3,763,200 elements
- Actual tensor size: 752,640 elements
- If we assume the first two dimensions are correct (25 × 196 = 4,900), then:
  - 752,640 ÷ 4,900 = 153.6 ≈ 153 channels
  - Expected channels: 12 × 64 = 768
  - Ratio: 768 ÷ 153 ≈ 5

This suggests the model is receiving 1/5 of the expected channels, possibly due to a window size or batch configuration issue.

## Files Involved

1. **Error Location**:
   - `/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/slam3r/blocks/multiview_blocks.py:90`
   - Function: `batched_cross_attn()`
   - Line: `qs = cross_attn.projq(xs_normed).reshape(Vx*B,Nx,num_heads, C//num_heads).permute(0, 2, 1, 3)`

2. **Call Stack**:
   - `slam3r_processor.py:779` - `l2w_inference()` call
   - `slam3r/utils/recon_utils.py:333` - `l2w_model()` forward pass
   - `slam3r/models.py:593` - `_decode_multiview()` method
   - `slam3r/models.py:333` - Multiview block forward pass

3. **Configuration Files**:
   - `/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/configs/wild.yaml` - May contain window size settings
   - Model checkpoint files that define architecture parameters

## How to Reproduce

### Prerequisites
1. Docker containers running:
   ```bash
   docker ps | grep -E "slam3r|mesh_service"
   ```

2. Ensure SLAM3R has the fix for 'hv' not defined error (already applied)

### Steps to Reproduce

1. **Start the services** (if not already running):
   ```bash
   docker compose up slam3r mesh_service rabbitmq -d
   ```

2. **Run the test script**:
   ```bash
   cd /home/sam3/Desktop/Toms_Workspace/WorldSystem
   /home/sam3/anaconda3/envs/3dreconstruction/bin/python test_slam3r_real_video.py
   ```

   This script:
   - Loads real drone video from `/data/20250617_211214/mav0/video_segments/20250617_211214_segment_1.mp4`
   - Sends 300 frames at realistic frame rate (14.7 fps)
   - Triggers SLAM3R bootstrap (frames 1-5)
   - Causes tensor reshape error on frame 6

3. **Monitor the logs**:
   ```bash
   # In one terminal - watch SLAM3R logs
   docker logs -f slam3r | grep -E "Bootstrap complete|ERROR|shape"
   
   # In another terminal - watch for specific error
   docker logs slam3r --tail 100 | grep "shape '\[25, 196, 12, 64\]'"
   ```

### Expected Output

1. **Bootstrap Success** (frames 1-5):
   ```
   2025-07-10 23:55:06,793  INFO  Bootstrap complete with 5 keyframes.
   2025-07-10 23:55:07,159  INFO  Published keyframe bootstrap_0 to mesh service (47681 points)
   2025-07-10 23:55:07,159  INFO  Published keyframe bootstrap_1 to mesh service (47671 points)
   ...
   ```

2. **Tensor Reshape Error** (frame 6+):
   ```
   RuntimeError: shape '[25, 196, 12, 64]' is invalid for input of size 752640
   ```

## Alternative Test Methods

### Using Synthetic Data (Does NOT reproduce error)
```bash
/home/sam3/anaconda3/envs/3dreconstruction/bin/python test_slam3r_keyframes.py
```
This sends synthetic frames with moving patterns but does not trigger the tensor reshape error.

### Direct Shared Memory Test (Bypasses SLAM3R)
```bash
/home/sam3/anaconda3/envs/3dreconstruction/bin/python test_shared_memory_direct.py
```
This tests the IPC mechanism directly without going through SLAM3R.

## Debug Information

To gather more debug information, you can:

1. **Check model configuration**:
   ```bash
   docker exec slam3r cat /app/SLAM3R_engine/configs/wild.yaml | grep -E "window|batch|stride"
   ```

2. **Inspect tensor dimensions** by adding debug prints to `multiview_blocks.py`:
   ```python
   print(f"DEBUG: xs shape={xs.shape}, Vx={Vx}, B={B}, Nx={Nx}, C={C}")
   print(f"DEBUG: xs_normed shape={xs_normed.shape}")
   print(f"DEBUG: projq output shape={cross_attn.projq(xs_normed).shape}")
   ```

## Next Steps - Todo List

### High Priority - Fix Tensor Reshape Error

1. **Debug tensor dimensions in multiview attention**
   - Add logging to track tensor shapes through the attention layers
   - Identify where the channel dimension gets reduced from 768 to 153
   - Check if window size configuration affects channel count

2. **Investigate model architecture mismatch**
   - Verify the loaded model checkpoint matches expected architecture
   - Check if encoder/decoder dimensions are properly configured
   - Compare bootstrap vs incremental processing tensor shapes

3. **Fix the reshape operation**
   - Adjust the reshape parameters based on actual tensor dimensions
   - Consider if the issue is related to batch size or sequence length
   - Test with different window configurations

### Medium Priority - Complete Integration

4. **Implement RabbitMQ consumer in mesh service**
   - Replace polling with event-driven keyframe detection
   - Process shared memory segments when notified via RabbitMQ
   - Add proper cleanup after processing

5. **Fix mesh service minimum point threshold**
   - Lower threshold to handle smaller point clouds
   - Add fallback for keyframes with few points
   - Test with varying point cloud sizes

6. **Add comprehensive logging**
   - Track tensor shapes throughout the pipeline
   - Log keyframe creation and publishing
   - Monitor shared memory usage and cleanup

### Low Priority - Optimization

7. **Performance benchmarking**
   - Test with full video sequences
   - Measure end-to-end latency
   - Profile GPU memory usage

8. **Integration testing**
   - Create automated tests for the full pipeline
   - Test error recovery mechanisms
   - Validate mesh quality with different inputs

The most critical task is fixing the tensor reshape error, as it blocks all incremental processing after bootstrap. Once fixed, SLAM3R should be able to process entire video sequences and generate continuous keyframes for mesh generation.