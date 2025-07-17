import numpy as np
import trimesh
import os
from glob import glob

# Find PLY files
ply_files = sorted(glob('/debug_output/slam3r_keyframe_*.ply'))
print(f'Found {len(ply_files)} PLY files')

# Analyze specific keyframes
keyframes_to_check = [10, 40, 80]
all_bounds = []

for kf in keyframes_to_check:
    filename = f'/debug_output/slam3r_keyframe_{kf:06d}.ply'
    if os.path.exists(filename):
        mesh = trimesh.load(filename)
        points = np.array(mesh.vertices)
        
        print(f'\n=== Keyframe {kf} ===')
        print(f'File: {filename}')
        print(f'Number of points: {len(points)}')
        print(f'Bounds:')
        print(f'  X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]')
        print(f'  Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]')
        print(f'  Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]')
        
        all_bounds.append({
            'kf': kf,
            'min': points.min(axis=0),
            'max': points.max(axis=0),
            'points': len(points)
        })

# Check for overlaps
print('\n=== Checking for spatial overlaps ===')
for i in range(len(all_bounds)):
    for j in range(i+1, len(all_bounds)):
        b1, b2 = all_bounds[i], all_bounds[j]
        
        # Check if bounding boxes overlap
        overlap_x = not (b1['max'][0] < b2['min'][0] or b2['max'][0] < b1['min'][0])
        overlap_y = not (b1['max'][1] < b2['min'][1] or b2['max'][1] < b1['min'][1])
        overlap_z = not (b1['max'][2] < b2['min'][2] or b2['max'][2] < b1['min'][2])
        
        if overlap_x and overlap_y and overlap_z:
            print(f'Keyframes {b1["kf"]} and {b2["kf"]} have overlapping bounds')
        else:
            print(f'Keyframes {b1["kf"]} and {b2["kf"]} have NO overlap')