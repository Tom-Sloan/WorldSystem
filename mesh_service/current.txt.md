This is the log of the mesh_service docker container in <LOG>. 

You are an expert programmer and debugger tasked with improving the quality of the mesh service. The mesh service is a TSDF marching cubes algorithm and so it should be able to run in real time. Your objective is to figure out why it's performing poorly and come up with a plan on how we can:
A. Figure out what is making it perform poorly in more specific detail
B. Come up with a plan on how to improve the quality of the mesh service
C. Solve this issue

After you have come up with a plan, re-evaluate based on the files the validity of your plan. If you do not feel extremely confident with this plan, come up with a new plan and re-evaluate. Repeat this until you have a highly confident plan. You're able to examine in depth all of the files in this project. Spending as much time as you need to ensure that what you're doing is correct. You're able to ask questions if needed. You're able to spend as much time as you want. Ultrathink

Furthermore, you can read the files in the mesh services /debug_output. There you're able to see the mesh.ply files as well as the pointcloud.ply files and the TSDF slices. You can read the readme in the debug_output to learn more. When comparing the PLY of the mesh to the pointcloud, the mesh is a solid chunk with triangles going across the hallway. However, in the pointcloud you can see the actual hallway itself. Furthermore, every pointcloud that has a keyframe divisible by 10 has the colors properly created on the points.

I have included three images: two showing the mesh appearance where the colors aren't being properly applied and the triangles are disjointed with poor connections, and another showing the point cloud which displays a tunnel with correct coloring. I've also included the logs to show the output at various stages and processing times.

Remember, the output is in real time.

<IMAGES>
/home/sam3/Desktop/Toms_Workspace/WorldSystem/Screenshot 2025-07-16 at 18.40.25.png
/home/sam3/Desktop/Toms_Workspace/WorldSystem/Screenshot 2025-07-16 at 18.40.38.png
/home/sam3/Desktop/Toms_Workspace/WorldSystem/Screenshot 2025-07-16 at 18.46.42.png
/home/sam3/Desktop/Toms_Workspace/WorldSystem/Screenshot 2025-07-16 at 18.45.46.png
<IMAGES>

<LOG>
mesh_service  |   Truncation distance: 0.100
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.627,-0.904,5.859] -> voxel=[27.469,41.916,117.173], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.622,-0.910,5.864] -> voxel=[27.554,41.809,117.289], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.621,-0.918,5.870] -> voxel=[27.589,41.648,117.407], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.619,-0.923,5.883] -> voxel=[27.615,41.533,117.662], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.619,-0.933,5.892] -> voxel=[27.613,41.336,117.834], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TIMING] TSDF integration kernel: 2 ms
mesh_service  | [TIMING] Total TSDF integration: 2 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 2 ms
mesh_service  | [TIMING] TSDF volume check: 4 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1600000
mesh_service  |   Modified voxels: 1599227
mesh_service  |   Weighted voxels: 165023
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 330046 active voxels (safety margin from 165023 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 165023 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108288, weight=100.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108289, weight=100.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108290, weight=100.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 100928, weight=62.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 100929, weight=100.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 100930, weight=37.000, pos=6
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109647, weight=5.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109648, weight=100.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109649, weight=100.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109650, weight=100.000, pos=3
mesh_service  | [MC DEBUG] Found 165023 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
mesh_service  | [MC DEBUG] Classifying 165023 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 109647): coords=[47,70,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 109648): coords=[48,70,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 109649): coords=[49,70,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 109650): coords=[50,70,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 100928): coords=[48,61,12]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.059,0.047,-0.073,1.000,-0.013,0.016,-0.000,-0.049]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.047,-0.023,-0.080,-0.073,0.016,-0.012,-0.028,-0.000]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [-0.023,-0.073,0.020,-0.080,-0.012,-0.051,-0.059,-0.028]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [-0.073,1.000,1.000,0.020,-0.051,1.000,1.000,-0.059]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [0.098,0.097,0.097,0.098,0.091,0.056,0.031,0.085]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=212, num_verts=12
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=238, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=251, num_verts=3
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=145, num_verts=9
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=0, num_verts=0
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 0 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 165023
mesh_service  |   Active voxels with triangles: 132773
mesh_service  |   Total vertices to generate: 925587
mesh_service  |   Last vert scan: 925584, orig: 3
mesh_service  |   Last occupied scan: 132772, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 132773 active voxels with 925587 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 1600000
mesh_service  |   Active voxels with triangles: 132773
mesh_service  |   Total vertices needed: 925587
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 132773
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 132773
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=0, out_idx=0, global_voxel=109647, verts=12
mesh_service  | [EXTRACT] idx=1, out_idx=1, global_voxel=109648, verts=6
mesh_service  | [EXTRACT] idx=2, out_idx=2, global_voxel=109649, verts=3
mesh_service  | [EXTRACT] idx=3, out_idx=3, global_voxel=109650, verts=9
mesh_service  | [EXTRACT] idx=5, out_idx=4, global_voxel=100929, verts=6
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 132773
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 132773
mesh_service  |   Output size: 132773
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=12, scan=0
mesh_service  |   [1] verts=6, scan=12
mesh_service  |   [2] verts=3, scan=18
mesh_service  |   [3] verts=9, scan=21
mesh_service  |   [4] verts=6, scan=30
mesh_service  |   [5] verts=6, scan=36
mesh_service  |   [6] verts=6, scan=42
mesh_service  |   [7] verts=6, scan=48
mesh_service  |   [8] verts=6, scan=54
mesh_service  |   [9] verts=6, scan=60
mesh_service  |   Last element: verts=3, scan=925584, total=925587
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x781236c00000
mesh_service  |   d_compressed_global_voxels ptr: 0x78130b600000
mesh_service  |   d_compressed_verts_scan ptr: 0x78130b703800
mesh_service  |   d_vertex_buffer: 0x78130c000000
mesh_service  |   d_normal_buffer: 0x781306000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 80x100x200
mesh_service  |   origin: [-2.000, -3.000, 0.000]
mesh_service  |   num_active_voxels: 132773
mesh_service  |   voxel_size: 0.050
mesh_service  |   iso_value: 0.000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 2075, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=109647, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=109648, vertex_offset=12
mesh_service  | [KERNEL] Thread 2: voxel_idx=109649, vertex_offset=18
mesh_service  | [KERNEL] Thread 0: coords=[47,70,13], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 1: coords=[48,70,13], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 2: coords=[49,70,13], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 0: base_idx=109647, max_idx=1600000
mesh_service  | [KERNEL] Thread 1: base_idx=109648, max_idx=1600000
mesh_service  | [KERNEL] Thread 2: base_idx=109649, max_idx=1600000
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=212
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=238
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=251
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=12
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=18
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=12
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=18
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 0 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 5 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 20 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 2 ms
mesh_service  |   - TSDF check: 4 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 0 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 5 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 21 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 77 µs
mesh_service  | Mesh generation completed in 27ms: 925587 vertices, 308529 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 27 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~21 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47969
mesh_service  |     - Points/sec: 1776629.630
mesh_service  |     - Vertices generated: 925587
mesh_service  |     - Faces generated: 308529
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 925587 vertices, 308529 faces in 27ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 2776761
mesh_service  | [DEBUG] Extracting colors for 47969 points
mesh_service  | [DEBUG] Color extraction complete, size: 143907
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 925587 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [0.397, 0.550, 0.650]
mesh_service  | [DEBUG Rerun] Vertex 1: [0.400, 0.549, 0.700]
mesh_service  | [DEBUG Rerun] Vertex 2: [0.400, 0.520, 0.650]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 8 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 59 µs
mesh_service  | [TIMING] Total Rerun publishing: 8 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 38 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 111 - Total Time: 36 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 27 ms (75.000%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~21.600 ms (estimate)
mesh_service  |      - Other: ~5.400 ms (estimate)
mesh_service  |   2. Rerun Publishing: 8 ms (22.222%)
mesh_service  |   3. Cleanup: 0 ms (0.000%)
mesh_service  |   4. Other (metrics, etc): 1 ms
mesh_service  | Performance: 27.778 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 36 ms
mesh_service  | [TIMING] Total message handling: 36 ms
mesh_service  | Received message, size: 322 bytes
mesh_service  | [TIMING] msgpack unpacking: 9 µs
mesh_service  | Message type: 8
mesh_service  | Iterating through msgpack map:
mesh_service  |   Key: type, Value type: 5
mesh_service  |   Key: keyframe_id, Value type: 5
mesh_service  |   Key: timestamp_ns, Value type: 2
mesh_service  |   Key: pose_matrix, Value type: 7
mesh_service  |   Key: shm_key, Value type: 5
mesh_service  |   Key: point_count, Value type: 2
mesh_service  |   Key: bbox, Value type: 7
mesh_service  | [DEBUG] Message has 'type' field: keyframe.new
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_112, keyframe_id: 112, type: keyframe.new, point_count: 47980
mesh_service  | [TIMING] Message parsing: 19 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 112 via RabbitMQ, shm_key: /slam3r_keyframe_112
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_112
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_112
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 43 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 15 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 47980, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Header pose_matrix - camera position: [0.000, 0.000, 0.000]
mesh_service  | [SHM DEBUG] Header bbox: [-0.778, -3.035, 5.442] to [2.175, 0.875, 19.495]
mesh_service  | [SHM DEBUG] Calculated total size: 719804
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x78132c02f000
mesh_service  | [TIMING] mmap full: 6 µs
mesh_service  | [TIMING] Total SharedMemory open: 168 µs (0.168 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 47980 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [TIMING] Spatial deduplication check: 214 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.000 m/s, points=47980
mesh_service  | [TIMING] Point/color data access: 1 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 47980 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [-0.625, -0.900, 5.854]
mesh_service  |   Point 1: [-0.621, -0.905, 5.860]
mesh_service  |   Point 2: [-0.619, -0.913, 5.867]
mesh_service  |   Point 3: [-0.618, -0.918, 5.879]
mesh_service  |   Point 4: [-0.618, -0.928, 5.887]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [-0.778, -3.035, 5.442]
mesh_service  |   Max: [2.175, 0.875, 19.495]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [-0.778, -3.035, 5.442]
mesh_service  |   Max: [2.175, 0.875, 19.495]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 280 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.549 MB)
mesh_service  | [TIMING] Octree update: 0 ms
mesh_service  | [CAMERA NORMAL PROVIDER] Computing normals for 47980 points
mesh_service  | [CAMERA NORMAL] Point 0: pos=[-0.625,-0.900,5.854], normal=[-0.105,-0.151,0.983], dist=5.956
mesh_service  | [CAMERA NORMAL] Point 1: pos=[-0.621,-0.905,5.860], normal=[-0.104,-0.152,0.983], dist=5.962
mesh_service  | [CAMERA NORMAL] Point 2: pos=[-0.619,-0.913,5.867], normal=[-0.104,-0.153,0.983], dist=5.969
mesh_service  | [CAMERA NORMAL] Point 3: pos=[-0.618,-0.918,5.879], normal=[-0.103,-0.154,0.983], dist=5.982
mesh_service  | [CAMERA NORMAL] Point 4: pos=[-0.618,-0.928,5.887], normal=[-0.103,-0.155,0.983], dist=5.992
mesh_service  | [CAMERA NORMAL] Point 5: pos=[-0.619,-0.936,5.906], normal=[-0.103,-0.156,0.982], dist=6.011
mesh_service  | [CAMERA NORMAL] Point 6: pos=[-0.618,-0.945,5.915], normal=[-0.103,-0.157,0.982], dist=6.022
mesh_service  | [CAMERA NORMAL] Point 7: pos=[-0.617,-0.954,5.928], normal=[-0.102,-0.158,0.982], dist=6.036
mesh_service  | [CAMERA NORMAL] Point 8: pos=[-0.618,-0.964,5.945], normal=[-0.102,-0.159,0.982], dist=6.054
mesh_service  | [CAMERA NORMAL] Point 9: pos=[-0.616,-0.974,5.953], normal=[-0.102,-0.161,0.982], dist=6.064
mesh_service  | [NORMAL ESTIMATION] Provider: 1 (Camera-based (fast))
mesh_service  | [NORMAL ESTIMATION] Time: 0 ms
mesh_service  | [NORMAL ESTIMATION] Points: 47980
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.000 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 47980, Complexity: 0.480
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 47980 points
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Camera position: [0.000, 0.000, 0.000]
mesh_service  | [TSDF DEBUG] integrate() called with 47980 points
mesh_service  | [TSDF DEBUG] Normal provider: External normals provided
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.100 m
mesh_service  | [TIMING] TSDF debug copy: 26 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [-0.625, -0.900, 5.854]
mesh_service  |   Point 1: [-0.621, -0.905, 5.860]
mesh_service  |   Point 2: [-0.619, -0.913, 5.867]
mesh_service  |   Point 3: [-0.618, -0.918, 5.879]
mesh_service  |   Point 4: [-0.618, -0.928, 5.887]
mesh_service  |   Point 5: [-0.619, -0.936, 5.906]
mesh_service  |   Point 6: [-0.618, -0.945, 5.915]
mesh_service  |   Point 7: [-0.617, -0.954, 5.928]
mesh_service  |   Point 8: [-0.618, -0.964, 5.945]
mesh_service  |   Point 9: [-0.616, -0.974, 5.953]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.000, Translation X: 40.000
mesh_service  |   Scale Y: 20.000, Translation Y: 60.000
mesh_service  |   Scale Z: 20.000, Translation Z: -0.000
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  |   Truncation distance: 0.100
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.625,-0.900,5.854] -> voxel=[27.498,42.009,117.081], normal=[-0.105,-0.151,0.983]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.621,-0.905,5.860] -> voxel=[27.581,41.904,117.207], normal=[-0.104,-0.152,0.983]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.619,-0.913,5.867] -> voxel=[27.611,41.743,117.330], normal=[-0.104,-0.153,0.983]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.618,-0.918,5.879] -> voxel=[27.639,41.631,117.575], normal=[-0.103,-0.154,0.983]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.618,-0.928,5.887] -> voxel=[27.636,41.432,117.737], normal=[-0.103,-0.155,0.983]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[25,40,115], max=[30,45,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1600000
mesh_service  |   Modified voxels: 1599204
mesh_service  |   Weighted voxels: 165344
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 330688 active voxels (safety margin from 165344 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 165344 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108128, weight=100.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108129, weight=100.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 116768, weight=100.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 116769, weight=100.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 116770, weight=100.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 116771, weight=52.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 101008, weight=49.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 101009, weight=100.000, pos=6
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 101010, weight=21.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 116767, weight=74.000, pos=0
mesh_service  | [MC DEBUG] Found 165344 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
mesh_service  | [MC DEBUG] Classifying 165344 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 116767): coords=[47,59,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 116768): coords=[48,59,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 116769): coords=[49,59,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 116770): coords=[50,59,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 116771): coords=[51,59,14]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.083,0.020,0.020,0.089,0.058,0.050,0.044,0.067]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.020,0.025,0.022,0.020,0.050,0.045,0.035,0.044]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [0.025,-0.053,-0.043,0.022,0.045,-0.046,-0.034,0.035]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [-0.053,-0.074,-0.058,-0.043,-0.046,-0.069,-0.092,-0.034]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [-0.074,1.000,1.000,-0.058,-0.069,1.000,1.000,-0.092]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=102, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=255, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=153, num_verts=6
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 0 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 165344
mesh_service  |   Active voxels with triangles: 133308
mesh_service  |   Total vertices to generate: 930267
mesh_service  |   Last vert scan: 930261, orig: 6
mesh_service  |   Last occupied scan: 133307, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 133308 active voxels with 930267 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 1600000
mesh_service  |   Active voxels with triangles: 133308
mesh_service  |   Total vertices needed: 930267
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 133308
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 133308
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=2, out_idx=0, global_voxel=116769, verts=6
mesh_service  | [EXTRACT] idx=4, out_idx=1, global_voxel=116771, verts=6
mesh_service  | [EXTRACT] idx=6, out_idx=2, global_voxel=101009, verts=6
mesh_service  | [EXTRACT] idx=7, out_idx=3, global_voxel=101010, verts=6
mesh_service  | [EXTRACT] idx=9, out_idx=4, global_voxel=108129, verts=6
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 133308
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 133308
mesh_service  |   Output size: 133308
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=6, scan=0
mesh_service  |   [1] verts=6, scan=6
mesh_service  |   [2] verts=6, scan=12
mesh_service  |   [3] verts=6, scan=18
mesh_service  |   [4] verts=6, scan=24
mesh_service  |   [5] verts=6, scan=30
mesh_service  |   [6] verts=9, scan=36
mesh_service  |   [7] verts=6, scan=45
mesh_service  |   [8] verts=9, scan=51
mesh_service  |   [9] verts=3, scan=60
mesh_service  |   Last element: verts=6, scan=930261, total=930267
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x781236c00000
mesh_service  |   d_compressed_global_voxels ptr: 0x78130b600000
mesh_service  |   d_compressed_verts_scan ptr: 0x78130b704800
mesh_service  |   d_vertex_buffer: 0x78130c000000
mesh_service  |   d_normal_buffer: 0x781306000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 80x100x200
mesh_service  |   origin: [-2.000, -3.000, 0.000]
mesh_service  |   num_active_voxels: 133308
mesh_service  |   voxel_size: 0.050
mesh_service  |   iso_value: 0.000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 2083, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=116769, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=116771, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: voxel_idx=101009, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: coords=[49,59,14], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 1: coords=[51,59,14], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 2: coords=[49,62,12], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 0: base_idx=116769, max_idx=1600000
mesh_service  | [KERNEL] Thread 1: base_idx=116771, max_idx=1600000
mesh_service  | [KERNEL] Thread 2: base_idx=101009, max_idx=1600000
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=102
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=153
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=102
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=6
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=12
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 0 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 7 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 9 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 0 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 7 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 10 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 57 µs
mesh_service  | Mesh generation completed in 14ms: 930267 vertices, 310089 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 14 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~10 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47980
mesh_service  |     - Points/sec: 3427142.857
mesh_service  |     - Vertices generated: 930267
mesh_service  |     - Faces generated: 310089
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 930267 vertices, 310089 faces in 14ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 2790801
mesh_service  | [DEBUG] Extracting colors for 47980 points
mesh_service  | [DEBUG] Color extraction complete, size: 143940
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 930267 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [0.466, -0.050, 0.700]
mesh_service  | [DEBUG Rerun] Vertex 1: [0.467, 0.000, 0.700]
mesh_service  | [DEBUG Rerun] Vertex 2: [0.475, -0.050, 0.750]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 7 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 50 µs
mesh_service  | [TIMING] Total Rerun publishing: 8 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 23 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 112 - Total Time: 22 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 14 ms (63.636%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~11.200 ms (estimate)
mesh_service  |      - Other: ~2.800 ms (estimate)
mesh_service  |   2. Rerun Publishing: 8 ms (36.364%)
mesh_service  |   3. Cleanup: 0 ms (0.000%)
mesh_service  |   4. Other (metrics, etc): 0 ms
mesh_service  | Performance: 45.455 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 22 ms
mesh_service  | [TIMING] Total message handling: 22 ms
mesh_service  | Received message, size: 322 bytes
mesh_service  | [TIMING] msgpack unpacking: 5 µs
mesh_service  | Message type: 8
mesh_service  | Iterating through msgpack map:
mesh_service  |   Key: type, Value type: 5
mesh_service  |   Key: keyframe_id, Value type: 5
mesh_service  |   Key: timestamp_ns, Value type: 2
mesh_service  |   Key: pose_matrix, Value type: 7
mesh_service  |   Key: shm_key, Value type: 5
mesh_service  |   Key: point_count, Value type: 2
mesh_service  |   Key: bbox, Value type: 7
mesh_service  | [DEBUG] Message has 'type' field: keyframe.new
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_113, keyframe_id: 113, type: keyframe.new, point_count: 47968
mesh_service  | [TIMING] Message parsing: 6 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 113 via RabbitMQ, shm_key: /slam3r_keyframe_113
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_113
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_113
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 25 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 8 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 47968, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Header pose_matrix - camera position: [0.000, 0.000, 0.000]
mesh_service  | [SHM DEBUG] Header bbox: [-0.776, -3.086, 5.441] to [2.221, 0.878, 19.568]
mesh_service  | [SHM DEBUG] Calculated total size: 719624
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x78132c02f000
mesh_service  | [TIMING] mmap full: 3 µs
mesh_service  | [TIMING] Total SharedMemory open: 72 µs (0.072 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 47968 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [TIMING] Spatial deduplication check: 147 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.000 m/s, points=47968
mesh_service  | [TIMING] Point/color data access: 0 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 47968 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [-0.625, -0.902, 5.867]
mesh_service  |   Point 1: [-0.621, -0.907, 5.873]
mesh_service  |   Point 2: [-0.619, -0.915, 5.879]
mesh_service  |   Point 3: [-0.618, -0.921, 5.891]
mesh_service  |   Point 4: [-0.618, -0.930, 5.900]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [-0.776, -3.086, 5.441]
mesh_service  |   Max: [2.221, 0.878, 19.568]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [-0.776, -3.086, 5.441]
mesh_service  |   Max: [2.221, 0.878, 19.568]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 124 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.549 MB)
mesh_service  | [TIMING] Octree update: 0 ms
mesh_service  | [CAMERA NORMAL PROVIDER] Computing normals for 47968 points
mesh_service  | [CAMERA NORMAL PROVIDER] Kernel launch failed: invalid configuration argument
mesh_service  | [NORMAL ESTIMATION] Failed, falling back to camera-based in TSDF
mesh_service  | [CAMERA NORMAL] Point 0: pos=[-0.625,-0.902,5.867], normal=[-0.105,-0.151,0.983], dist=5.969
mesh_service  | [CAMERA NORMAL] Point 1: pos=[-0.621,-0.907,5.873], normal=[-0.104,-0.152,0.983], dist=5.975
mesh_service  | [CAMERA NORMAL] Point 2: pos=[-0.619,-0.915,5.879], normal=[-0.104,-0.153,0.983], dist=5.982
mesh_service  | [CAMERA NORMAL] Point 3: pos=[-0.618,-0.921,5.891], normal=[-0.103,-0.154,0.983], dist=5.995
mesh_service  | [CAMERA NORMAL] Point 4: pos=[-0.618,-0.930,5.900], normal=[-0.103,-0.155,0.983], dist=6.004
mesh_service  | [CAMERA NORMAL] Point 5: pos=[-0.619,-0.938,5.918], normal=[-0.103,-0.156,0.982], dist=6.024
mesh_service  | [CAMERA NORMAL] Point 6: pos=[-0.618,-0.947,5.928], normal=[-0.102,-0.157,0.982], dist=6.034
mesh_service  | [CAMERA NORMAL] Point 7: pos=[-0.617,-0.956,5.941], normal=[-0.102,-0.158,0.982], dist=6.049
mesh_service  | [CAMERA NORMAL] Point 8: pos=[-0.618,-0.966,5.957], normal=[-0.102,-0.159,0.982], dist=6.067
mesh_service  | [CAMERA NORMAL] Point 9: pos=[-0.616,-0.976,5.966], normal=[-0.101,-0.161,0.982], dist=6.076
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.000 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 47968, Complexity: 0.480
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 47968 points
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Camera position: [0.000, 0.000, 0.000]
mesh_service  | [TSDF DEBUG] integrate() called with 47968 points
mesh_service  | [TSDF DEBUG] Normal provider: Camera-based fallback (improved carving)
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.100 m
mesh_service  | [TIMING] TSDF debug copy: 10 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [-0.625, -0.902, 5.867]
mesh_service  |   Point 1: [-0.621, -0.907, 5.873]
mesh_service  |   Point 2: [-0.619, -0.915, 5.879]
mesh_service  |   Point 3: [-0.618, -0.921, 5.891]
mesh_service  |   Point 4: [-0.618, -0.930, 5.900]
mesh_service  |   Point 5: [-0.619, -0.938, 5.918]
mesh_service  |   Point 6: [-0.618, -0.947, 5.928]
mesh_service  |   Point 7: [-0.617, -0.956, 5.941]
mesh_service  |   Point 8: [-0.618, -0.966, 5.957]
mesh_service  |   Point 9: [-0.616, -0.976, 5.966]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.000, Translation X: 40.000
mesh_service  |   Scale Y: 20.000, Translation Y: 60.000
mesh_service  |   Scale Z: 20.000, Translation Z: -0.000
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  |   Truncation distance: 0.100
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.625,-0.902,5.867] -> voxel=[27.499,41.963,117.341], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.621,-0.907,5.873] -> voxel=[27.585,41.857,117.464], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.619,-0.915,5.879] -> voxel=[27.614,41.698,117.585], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.618,-0.921,5.891] -> voxel=[27.642,41.586,117.830], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.618,-0.930,5.900] -> voxel=[27.639,41.390,117.990], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[25,39,115], max=[30,44,120], volume_dims=[80,100,200]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1600000
mesh_service  |   Modified voxels: 1599211
mesh_service  |   Weighted voxels: 165816
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 331632 active voxels (safety margin from 165816 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 165816 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109647, weight=5.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109648, weight=100.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 109649, weight=100.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108528, weight=100.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108529, weight=100.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108530, weight=100.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108367, weight=100.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108368, weight=100.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108369, weight=100.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108370, weight=100.000, pos=6
mesh_service  | [MC DEBUG] Found 165816 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
mesh_service  | [MC DEBUG] Classifying 165816 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 108528): coords=[48,56,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 108529): coords=[49,56,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 108530): coords=[50,56,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 108367): coords=[47,54,13]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 108368): coords=[48,54,13]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.084,0.018,0.042,0.084,0.019,0.033,0.033,0.052]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.018,-0.079,-0.090,0.042,0.033,-0.012,-0.051,0.033]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [-0.079,1.000,1.000,-0.090,-0.012,-0.044,-0.060,-0.051]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [0.093,0.079,0.065,0.098,0.057,0.058,0.036,0.052]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [0.079,0.030,0.031,0.065,0.058,0.038,0.033,0.036]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=102, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=249, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=0, num_verts=0
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 1 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 165816
mesh_service  |   Active voxels with triangles: 133549
mesh_service  |   Total vertices to generate: 932499
mesh_service  |   Last vert scan: 932496, orig: 3
mesh_service  |   Last occupied scan: 133548, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 133549 active voxels with 932499 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 1600000
mesh_service  |   Active voxels with triangles: 133549
mesh_service  |   Total vertices needed: 932499
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 133549
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 133549
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=1, out_idx=0, global_voxel=108529, verts=6
mesh_service  | [EXTRACT] idx=2, out_idx=1, global_voxel=108530, verts=6
mesh_service  | [EXTRACT] idx=5, out_idx=2, global_voxel=108369, verts=6
mesh_service  | [EXTRACT] idx=6, out_idx=3, global_voxel=108370, verts=6
mesh_service  | [EXTRACT] idx=7, out_idx=4, global_voxel=109647, verts=12
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 133549
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 133549
mesh_service  |   Output size: 133549
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=6, scan=0
mesh_service  |   [1] verts=6, scan=6
mesh_service  |   [2] verts=6, scan=12
mesh_service  |   [3] verts=6, scan=18
mesh_service  |   [4] verts=12, scan=24
mesh_service  |   [5] verts=6, scan=36
mesh_service  |   [6] verts=3, scan=42
mesh_service  |   [7] verts=9, scan=45
mesh_service  |   [8] verts=6, scan=54
mesh_service  |   [9] verts=6, scan=60
mesh_service  |   Last element: verts=3, scan=932496, total=932499
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x781236c00000
mesh_service  |   d_compressed_global_voxels ptr: 0x78130b600000
mesh_service  |   d_compressed_verts_scan ptr: 0x78130b705000
mesh_service  |   d_vertex_buffer: 0x78130c000000
mesh_service  |   d_normal_buffer: 0x781306000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 80x100x200
mesh_service  |   origin: [-2.000, -3.000, 0.000]
mesh_service  |   num_active_voxels: 133549
mesh_service  |   voxel_size: 0.050
mesh_service  |   iso_value: 0.000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 2087, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=108529, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=108530, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: voxel_idx=108369, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: coords=[49,56,13], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 1: coords=[50,56,13], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 2: coords=[49,54,13], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 0: base_idx=108529, max_idx=1600000
mesh_service  | [KERNEL] Thread 1: base_idx=108530, max_idx=1600000
mesh_service  | [KERNEL] Thread 2: base_idx=108369, max_idx=1600000
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=102
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=249
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=102
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=6
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=12
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 1 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 6 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 15 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 1 ms
mesh_service  |   - Triangle gen: 1 ms
mesh_service  |   - Output copy: 6 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 16 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 65 µs
mesh_service  | Mesh generation completed in 18ms: 932499 vertices, 310833 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 18 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~16 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47968
mesh_service  |     - Points/sec: 2664888.889
mesh_service  |     - Vertices generated: 932499
mesh_service  |     - Faces generated: 310833
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 932499 vertices, 310833 faces in 18ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 2797497
mesh_service  | [DEBUG] Extracting colors for 47968 points
mesh_service  | [DEBUG] Color extraction complete, size: 143904
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 932499 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [0.459, -0.200, 0.650]
mesh_service  | [DEBUG Rerun] Vertex 1: [0.466, -0.150, 0.650]
mesh_service  | [DEBUG Rerun] Vertex 2: [0.487, -0.200, 0.700]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 6 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 200 µs
mesh_service  | [TIMING] Total Rerun publishing: 7 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 19 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 113 - Total Time: 25 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 18 ms (72.000%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~14.400 ms (estimate)
mesh_service  |      - Other: ~3.600 ms (estimate)
mesh_service  |   2. Rerun Publishing: 7 ms (28.000%)
mesh_service  |   3. Cleanup: 0 ms (0.000%)
mesh_service  |   4. Other (metrics, etc): 0 ms
mesh_service  | Performance: 40.000 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 25 ms
mesh_service  | [TIMING] Total message handling: 25 ms


w Enable Watch
<LOG>