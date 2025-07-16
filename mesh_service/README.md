# Mesh Service Debug Output

This directory contains debug output from the mesh service. Files are saved every 10 frames.

## Files Generated

1. **pointcloud_XXXXXX.ply** - Raw point clouds from SLAM3R
   - Check orientation: X should span ~25m (hallway), Y ~15m (atrium), Z ~7m (height)
   - View with: `meshlab pointcloud_*.ply`

2. **tsdf_slice_XXXXXX.txt** - TSDF volume slices at camera height
   - Format: X Y TSDF_value
   - Visualize with: `gnuplot -e "set view map; splot 'tsdf_slice_000010.txt' with image" -`

3. **mesh_XXXXXX.ply** - Generated meshes from marching cubes
   - Check for proper geometry and scale
   - View with: `meshlab mesh_*.ply`

## What to Check

1. **Point Cloud Bounds**: Should roughly match scene dimensions
   - X: ~25 meters (hallway length)
   - Y: ~15 meters (atrium width)
   - Z: ~7 meters (height)

2. **TSDF Values**: Should show surface boundaries
   - Negative values inside objects
   - Positive values outside objects
   - Zero crossing at surfaces

3. **Mesh Quality**: 
   - No excessive vertices (millions)
   - Proper scale matching point clouds
   - Smooth surfaces without holes

## Debugging Tips

- If no files appear, check docker logs: `docker logs mesh_service`
- Look for `[DEBUG SAVE]` messages in logs
- Check volume mount: `docker exec mesh_service ls /debug_output`