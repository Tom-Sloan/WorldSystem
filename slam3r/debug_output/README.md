# SLAM3R Debug Output

This directory contains debug output from the SLAM3R service. Point clouds are saved every 10 keyframes.

## Files Generated

1. **slam3r_keyframe_XXXXXX.ply** - Filtered point clouds from SLAM3R
   - Contains 3D points that pass confidence threshold
   - Colored by RGB from original image or confidence gradient (red=low, green=high)
   - Includes metadata: confidence stats, keyframe ID, point count
   - View with: `meshlab slam3r_keyframe_*.ply`

## What to Check

1. **Point Cloud Quality**: 
   - Points should have reasonable confidence values
   - No extreme outliers (filtered by 100m bounds check)
   - Color mapping should correspond to image content

2. **Confidence Values**: 
   - Check metadata in PLY for min/max/mean confidence
   - Low confidence may indicate poor reconstruction quality
   - Threshold is adaptively set (default 12.0, fallback 6.0)

3. **Spatial Distribution**: 
   - Points should roughly match scene geometry
   - Check for gaps or missing regions
   - Verify scale matches expected scene dimensions

## Debugging Tips

- If no files appear, check docker logs: `docker logs slam3r`
- Look for `[DEBUG] Saved point cloud` messages in logs
- Check volume mount in docker-compose.yml
- Verify confidence threshold isn't filtering all points