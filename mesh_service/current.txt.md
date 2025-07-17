This is the log of the mesh_service docker container in <LOG>. This is the log of the slam3r docker container in <LOG SLAM3R>. 

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
mesh_service  | [CAMERA NORMAL] Point 4: pos=[-0.705,-0.716,1.075], normal=[-0.465,-0.493,0.735], dist=1.493
mesh_service  | [CAMERA NORMAL] Point 5: pos=[-0.704,-0.722,1.085], normal=[-0.461,-0.494,0.737], dist=1.502
mesh_service  | [CAMERA NORMAL] Point 6: pos=[-0.703,-0.730,1.093], normal=[-0.457,-0.496,0.738], dist=1.512
mesh_service  | [CAMERA NORMAL] Point 7: pos=[-0.702,-0.737,1.103], normal=[-0.454,-0.497,0.739], dist=1.522
mesh_service  | [CAMERA NORMAL] Point 8: pos=[-0.702,-0.743,1.112], normal=[-0.451,-0.499,0.740], dist=1.532
mesh_service  | [CAMERA NORMAL] Point 9: pos=[-0.701,-0.751,1.122], normal=[-0.447,-0.500,0.742], dist=1.543
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.0437 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 47179, Complexity: 0.4718
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 47179 points
mesh_service  | [MC DEBUG] Camera pose received (expecting row-major):
mesh_service  |   [0-3]: 0.9999, 0.0064, -0.0089, 0.0000
mesh_service  |   [4-7]: -0.0064, 1.0000, 0.0053, 0.0000
mesh_service  |   [8-11]: 0.0089, -0.0052, 0.9999, 0.0000
mesh_service  |   [12-15]: -0.0113, 0.0204, -0.0225, 1.0000
mesh_service  | [MC DEBUG] Camera position should be at [12,13,14]: [-0.0113, 0.0204, -0.0225]
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Received camera pose pointer: 0x767528530010
mesh_service  | [TSDF DEBUG] Interpreting pose matrix as ROW-MAJOR:
mesh_service  |   [    0.9999,     0.0064,    -0.0089,     0.0000]
mesh_service  |   [   -0.0064,     1.0000,     0.0053,     0.0000]
mesh_service  |   [    0.0089,    -0.0052,     0.9999,     0.0000]
mesh_service  |   [   -0.0113,     0.0204,    -0.0225,     1.0000]
mesh_service  | [TSDF DEBUG] Translation option 1 (indices 3,7,11): [0.0000, 0.0000, 0.0000]
mesh_service  | [TSDF DEBUG] Translation option 2 (indices 12,13,14): [-0.0113, 0.0204, -0.0225]
mesh_service  | [TSDF DEBUG] Using option 2 (row-major) as camera position: [-0.0113, 0.0204, -0.0225]
mesh_service  | [TSDF DEBUG] Rotation matrix determinant (should be ~1): 1.0000
mesh_service  | [TSDF DEBUG] integrate() called with 47179 points
mesh_service  | [TSDF DEBUG] Normal provider: Camera-based fallback (improved carving)
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.1000 m
mesh_service  | [TIMING] TSDF debug copy: 9 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [-0.7079, -0.6913, 1.0410]
mesh_service  |   Point 1: [-0.7064, -0.6970, 1.0486]
mesh_service  |   Point 2: [-0.7057, -0.7039, 1.0570]
mesh_service  |   Point 3: [-0.7059, -0.7101, 1.0661]
mesh_service  |   Point 4: [-0.7054, -0.7165, 1.0752]
mesh_service  |   Point 5: [-0.7038, -0.7224, 1.0847]
mesh_service  |   Point 6: [-0.7027, -0.7298, 1.0933]
mesh_service  |   Point 7: [-0.7023, -0.7368, 1.1028]
mesh_service  |   Point 8: [-0.7016, -0.7434, 1.1118]
mesh_service  |   Point 9: [-0.7014, -0.7509, 1.1220]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.0000, Translation X: 18.2258
mesh_service  |   Scale Y: 20.0000, Translation Y: 42.9302
mesh_service  |   Scale Z: 20.0000, Translation Z: -9.8525
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 33x42x85
mesh_service  |   Origin: [-0.9113, -2.1465, 0.4926]
mesh_service  |   Voxel size: 0.0500
mesh_service  |   Truncation distance: 0.1000
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.708,-0.691,1.041] -> voxel=[4.068,29.105,10.968], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.706,-0.697,1.049] -> voxel=[4.098,28.990,11.119], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.706,-0.704,1.057] -> voxel=[4.112,28.852,11.288], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.706,-0.710,1.066] -> voxel=[4.108,28.729,11.470], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.705,-0.716,1.075] -> voxel=[4.117,28.601,11.652], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[2,27,8], max=[7,32,13], volume_dims=[33,42,85]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,85]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,85]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,85]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,85]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 117810
mesh_service  |   Modified voxels: 117677
mesh_service  |   Weighted voxels: 21138
mesh_service  | [MC DEBUG] TSDF sign distribution (weighted voxels only):
mesh_service  |   Positive (empty): 0
mesh_service  |   Negative (occupied): 0
mesh_service  |   Near zero: 0
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 33x42x85
mesh_service  |   Origin: [-0.9113, -2.1465, 0.4926]
mesh_service  |   Voxel size: 0.0500
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 42276 active voxels (safety margin from 21138 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 21138 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22080, weight=40.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22081, weight=32.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22082, weight=16.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22016, weight=34.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22037, weight=22.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22038, weight=35.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22039, weight=29.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22040, weight=16.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22046, weight=12.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22047, weight=31.000, pos=6
mesh_service  | [MC DEBUG] Found 21138 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 
mesh_service  | [MC DEBUG] Classifying 21138 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 22016): coords=[5,37,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 22037): coords=[26,37,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 22038): coords=[27,37,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 22039): coords=[28,37,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 22040): coords=[29,37,15]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.074,1.000,1.000,0.060,0.037,1.000,1.000,0.040]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.044,0.042,0.030,0.041,0.057,0.038,0.028,0.032]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [0.042,0.051,0.011,0.030,0.038,0.019,0.015,0.028]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [0.051,-0.063,-0.055,0.011,0.019,-0.017,-0.029,0.015]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [-0.063,1.000,1.000,-0.055,-0.017,1.000,1.000,-0.029]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=102, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=153, num_verts=6
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 1 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 21138
mesh_service  |   Active voxels with triangles: 13772
mesh_service  |   Total vertices to generate: 89859
mesh_service  |   Last vert scan: 89847, orig: 12
mesh_service  |   Last occupied scan: 13771, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 13772 active voxels with 89859 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 27744000
mesh_service  |   Active voxels with triangles: 13772
mesh_service  |   Total vertices needed: 89859
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 13772
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 13772
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=3, out_idx=0, global_voxel=22039, verts=6
mesh_service  | [EXTRACT] idx=4, out_idx=1, global_voxel=22040, verts=6
mesh_service  | [EXTRACT] idx=5, out_idx=2, global_voxel=22046, verts=9
mesh_service  | [EXTRACT] idx=6, out_idx=3, global_voxel=22047, verts=3
mesh_service  | [EXTRACT] idx=7, out_idx=4, global_voxel=22080, verts=3
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 13772
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 13772
mesh_service  |   Output size: 13772
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=6, scan=0
mesh_service  |   [1] verts=6, scan=6
mesh_service  |   [2] verts=9, scan=12
mesh_service  |   [3] verts=3, scan=21
mesh_service  |   [4] verts=3, scan=24
mesh_service  |   [5] verts=9, scan=27
mesh_service  |   [6] verts=9, scan=36
mesh_service  |   [7] verts=9, scan=45
mesh_service  |   [8] verts=9, scan=54
mesh_service  |   [9] verts=9, scan=63
mesh_service  |   Last element: verts=12, scan=89847, total=89859
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x767416000000
mesh_service  |   d_compressed_global_voxels ptr: 0x767442a29e00
mesh_service  |   d_compressed_verts_scan ptr: 0x767442a44e00
mesh_service  |   d_vertex_buffer: 0x7673ea000000
mesh_service  |   d_normal_buffer: 0x7673e4000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 33x42x85
mesh_service  |   origin: [-0.9113, -2.1465, 0.4926]
mesh_service  |   num_active_voxels: 13772
mesh_service  |   voxel_size: 0.0500
mesh_service  |   iso_value: 0.0000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 216, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=22039, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=22040, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: voxel_idx=22046, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: coords=[28,37,15], dims=[33,42,85]
mesh_service  | [KERNEL] Thread 1: coords=[29,37,15], dims=[33,42,85]
mesh_service  | [KERNEL] Thread 2: coords=[2,38,15], dims=[33,42,85]
mesh_service  | [KERNEL] Thread 0: base_idx=22039, max_idx=117810
mesh_service  | [KERNEL] Thread 1: base_idx=22040, max_idx=117810
mesh_service  | [KERNEL] Thread 2: base_idx=22046, max_idx=117810
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=102
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=153
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=217
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
mesh_service  | [TIMING] Output copy (D2H): 2 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 9 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 1 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 2 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 9 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 50 µs
mesh_service  | Mesh generation completed in 12ms: 89859 vertices, 29953 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 12 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~9 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47179
mesh_service  |     - Points/sec: 3931583.3333
mesh_service  |     - Vertices generated: 89859
mesh_service  |     - Faces generated: 29953
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 89859 vertices, 29953 faces in 12ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 269577
mesh_service  | [DEBUG] Extracting colors for 47179 points
mesh_service  | [DEBUG] Color extraction complete, size: 141537
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 89859 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [0.5111, -0.2965, 1.2426]
mesh_service  | [DEBUG Rerun] Vertex 1: [0.4969, -0.2465, 1.2426]
mesh_service  | [DEBUG Rerun] Vertex 2: [0.5150, -0.2965, 1.2926]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 0 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 41 µs
mesh_service  | [TIMING] Total Rerun publishing: 0 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 24 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 27 - Total Time: 13 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 12 ms (92.3077%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~9.6000 ms (estimate)
mesh_service  |      - Other: ~2.4000 ms (estimate)
mesh_service  |   2. Rerun Publishing: 0 ms (0.0000%)
mesh_service  |   3. Cleanup: 0 ms (0.0000%)
mesh_service  |   4. Other (metrics, etc): 1 ms
mesh_service  | Performance: 76.9231 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 13 ms
mesh_service  | [TIMING] Total message handling: 13 ms
mesh_service  | Received message, size: 320 bytes
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
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_28, keyframe_id: 28, type: keyframe.new, point_count: 47176
mesh_service  | [TIMING] Message parsing: 13 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 28 via RabbitMQ, shm_key: /slam3r_keyframe_28
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_28
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_28
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 13 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 7 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 47176, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Full pose matrix from shared memory (row-major):
mesh_service  |   [    0.9999,     0.0072,    -0.0151,     0.0000]
mesh_service  |   [   -0.0072,     1.0000,    -0.0000,     0.0000]
mesh_service  |   [    0.0151,     0.0001,     0.9999,     0.0000]
mesh_service  |   [   -0.0172,     0.0113,     0.0151,     1.0000]
mesh_service  | [SHM DEBUG] Camera position (translation) from pose: [-0.0172, 0.0113, 0.0151]
mesh_service  | [SHM DEBUG] Checking alternate indices [3,7,11] (column-major): [0.0000, 0.0000, 0.0000]
mesh_service  | [SHM DEBUG] Full pose matrix as linear array:
mesh_service  |   [0-3]: 0.9999, 0.0072, -0.0151, 0.0000
mesh_service  |   [4-7]: -0.0072, 1.0000, -0.0000, 0.0000
mesh_service  |   [8-11]: 0.0151, 0.0001, 0.9999, 0.0000
mesh_service  |   [12-15]: -0.0172, 0.0113, 0.0151, 1.0000
mesh_service  | [SHM DEBUG] Header bbox: [-0.7208, -2.0860, 0.7009] to [0.5565, 1.0182, 7.7885]
mesh_service  | [SHM DEBUG] Calculated total size: 707744
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x767528530000
mesh_service  | [TIMING] mmap full: 2 µs
mesh_service  | [TIMING] Total SharedMemory open: 76 µs (0.0760 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 47176 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [MESH GEN DEBUG] Received pose matrix from keyframe:
mesh_service  |   [    0.9999,     0.0072,    -0.0151,     0.0000]
mesh_service  |   [   -0.0072,     1.0000,    -0.0000,     0.0000]
mesh_service  |   [    0.0151,     0.0001,     0.9999,     0.0000]
mesh_service  |   [   -0.0172,     0.0113,     0.0151,     1.0000]
mesh_service  | [MESH GEN DEBUG] Camera position from pose: [-0.0172, 0.0113, 0.0151]
mesh_service  | [TIMING] Spatial deduplication check: 134 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.0582 m/s, points=47176
mesh_service  | [TIMING] Point/color data access: 0 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 47176 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [-0.7086, -0.6981, 1.0544]
mesh_service  |   Point 1: [-0.7073, -0.7040, 1.0618]
mesh_service  |   Point 2: [-0.7067, -0.7108, 1.0698]
mesh_service  |   Point 3: [-0.7070, -0.7169, 1.0786]
mesh_service  |   Point 4: [-0.7067, -0.7232, 1.0872]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [-0.7208, -2.0860, 0.7009]
mesh_service  |   Max: [0.5565, 1.0182, 7.7885]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [-0.7208, -2.0860, 0.7009]
mesh_service  |   Max: [0.5565, 1.0182, 7.7885]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 120 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.5399 MB)
mesh_service  | [TIMING] Octree update: 0 ms
mesh_service  | [CAMERA NORMAL PROVIDER] Computing normals for 47176 points with grid 185 x 1 x 1 and block 256
mesh_service  | [CAMERA NORMAL PROVIDER] Kernel launch failed: invalid configuration argument
mesh_service  | [NORMAL ESTIMATION] Failed, falling back to camera-based in TSDF
mesh_service  | [CAMERA NORMAL] Point 0: pos=[-0.709,-0.698,1.054], normal=[-0.482,-0.494,0.724], dist=1.436
mesh_service  | [CAMERA NORMAL] Point 1: pos=[-0.707,-0.704,1.062], normal=[-0.478,-0.496,0.725], dist=1.443
mesh_service  | [CAMERA NORMAL] Point 2: pos=[-0.707,-0.711,1.070], normal=[-0.475,-0.497,0.726], dist=1.452
mesh_service  | [CAMERA NORMAL] Point 3: pos=[-0.707,-0.717,1.079], normal=[-0.472,-0.498,0.727], dist=1.462
mesh_service  | [CAMERA NORMAL] Point 4: pos=[-0.707,-0.723,1.087], normal=[-0.469,-0.499,0.729], dist=1.471
mesh_service  | [CAMERA NORMAL] Point 5: pos=[-0.705,-0.729,1.096], normal=[-0.465,-0.500,0.730], dist=1.480
mesh_service  | [CAMERA NORMAL] Point 6: pos=[-0.704,-0.736,1.105], normal=[-0.461,-0.502,0.732], dist=1.489
mesh_service  | [CAMERA NORMAL] Point 7: pos=[-0.704,-0.743,1.114], normal=[-0.458,-0.503,0.733], dist=1.500
mesh_service  | [CAMERA NORMAL] Point 8: pos=[-0.703,-0.750,1.123], normal=[-0.455,-0.504,0.734], dist=1.509
mesh_service  | [CAMERA NORMAL] Point 9: pos=[-0.703,-0.757,1.133], normal=[-0.451,-0.506,0.735], dist=1.520
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.0582 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 47176, Complexity: 0.4718
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 47176 points
mesh_service  | [MC DEBUG] Camera pose received (expecting row-major):
mesh_service  |   [0-3]: 0.9999, 0.0072, -0.0151, 0.0000
mesh_service  |   [4-7]: -0.0072, 1.0000, -0.0000, 0.0000
mesh_service  |   [8-11]: 0.0151, 0.0001, 0.9999, 0.0000
mesh_service  |   [12-15]: -0.0172, 0.0113, 0.0151, 1.0000
mesh_service  | [MC DEBUG] Camera position should be at [12,13,14]: [-0.0172, 0.0113, 0.0151]
mesh_service  | [MC BOUNDS UPDATE] Point cloud bounds changed, re-initializing TSDF
mesh_service  |   Old bounds: [-0.9113,-2.1465,0.4926] to [0.7387,-0.0465,4.7426]
mesh_service  |   New bounds: [-0.9153,-2.1521,0.5010] to [0.7409,-0.0712,4.6705]
mesh_service  | [SIMPLE TSDF INIT] Initializing TSDF volume:
mesh_service  |   Volume min: [-0.9153, -2.1521, 0.5010]
mesh_service  |   Volume max: [0.7409, -0.0712, 4.6705]
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Volume dimensions: [33, 42, 83]
mesh_service  |   Total voxels: 115038
mesh_service  | SimpleTSDF initialized:
mesh_service  |   Volume dims: 33x42x83
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Memory usage: 0.8777 MB
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Received camera pose pointer: 0x767528530010
mesh_service  | [TSDF DEBUG] Interpreting pose matrix as ROW-MAJOR:
mesh_service  |   [    0.9999,     0.0072,    -0.0151,     0.0000]
mesh_service  |   [   -0.0072,     1.0000,    -0.0000,     0.0000]
mesh_service  |   [    0.0151,     0.0001,     0.9999,     0.0000]
mesh_service  |   [   -0.0172,     0.0113,     0.0151,     1.0000]
mesh_service  | [TSDF DEBUG] Translation option 1 (indices 3,7,11): [0.0000, 0.0000, 0.0000]
mesh_service  | [TSDF DEBUG] Translation option 2 (indices 12,13,14): [-0.0172, 0.0113, 0.0151]
mesh_service  | [TSDF DEBUG] Using option 2 (row-major) as camera position: [-0.0172, 0.0113, 0.0151]
mesh_service  | [TSDF DEBUG] Rotation matrix determinant (should be ~1): 1.0000
mesh_service  | [TSDF DEBUG] integrate() called with 47176 points
mesh_service  | [TSDF DEBUG] Normal provider: Camera-based fallback (improved carving)
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.1000 m
mesh_service  | [TIMING] TSDF debug copy: 6 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [-0.7086, -0.6981, 1.0544]
mesh_service  |   Point 1: [-0.7073, -0.7040, 1.0618]
mesh_service  |   Point 2: [-0.7067, -0.7108, 1.0698]
mesh_service  |   Point 3: [-0.7070, -0.7169, 1.0786]
mesh_service  |   Point 4: [-0.7067, -0.7232, 1.0872]
mesh_service  |   Point 5: [-0.7053, -0.7291, 1.0962]
mesh_service  |   Point 6: [-0.7043, -0.7365, 1.1046]
mesh_service  |   Point 7: [-0.7041, -0.7433, 1.1141]
mesh_service  |   Point 8: [-0.7034, -0.7498, 1.1227]
mesh_service  |   Point 9: [-0.7032, -0.7573, 1.1326]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.0000, Translation X: 18.3063
mesh_service  |   Scale Y: 20.0000, Translation Y: 43.0426
mesh_service  |   Scale Z: 20.0000, Translation Z: -10.0203
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 33x42x83
mesh_service  |   Origin: [-0.9153, -2.1521, 0.5010]
mesh_service  |   Voxel size: 0.0500
mesh_service  |   Truncation distance: 0.1000
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.709,-0.698,1.054] -> voxel=[4.135,29.080,11.068], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.707,-0.704,1.062] -> voxel=[4.160,28.963,11.215], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.707,-0.711,1.070] -> voxel=[4.173,28.827,11.375], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.707,-0.717,1.079] -> voxel=[4.166,28.705,11.551], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.707,-0.723,1.087] -> voxel=[4.172,28.579,11.723], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[2,27,9], max=[7,32,14], volume_dims=[33,42,83]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,83]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,83]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,83]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,83]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 1 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 115038
mesh_service  |   Modified voxels: 114866
mesh_service  |   Weighted voxels: 19866
mesh_service  | [MC DEBUG] TSDF sign distribution (weighted voxels only):
mesh_service  |   Positive (empty): 0
mesh_service  |   Negative (occupied): 0
mesh_service  |   Near zero: 0
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 33x42x83
mesh_service  |   Origin: [-0.9153, -2.1521, 0.5010]
mesh_service  |   Voxel size: 0.0500
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 39732 active voxels (safety margin from 19866 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 19866 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22016, weight=9.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22037, weight=5.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22038, weight=12.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22048, weight=13.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22049, weight=8.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22070, weight=6.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22071, weight=15.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22072, weight=14.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22073, weight=4.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 22079, weight=5.000, pos=6
mesh_service  | [MC DEBUG] Found 19866 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 
mesh_service  | [MC DEBUG] Classifying 19866 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 22048): coords=[4,38,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 22049): coords=[5,38,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 22070): coords=[26,38,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 22071): coords=[27,38,15]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 22072): coords=[28,38,15]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.035,0.176,0.040,-0.009,0.039,0.043,0.288,0.029]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.176,1.000,1.000,0.040,0.043,1.000,1.000,0.288]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [-0.004,0.062,0.016,0.069,0.283,0.021,0.002,0.001]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [0.062,0.029,0.025,0.016,0.021,0.013,0.027,0.002]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [0.029,0.009,0.224,0.025,0.013,-0.050,-0.041,0.027]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=8, num_verts=3
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=1, num_verts=3
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=96, num_verts=6
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 1 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 19866
mesh_service  |   Active voxels with triangles: 14551
mesh_service  |   Total vertices to generate: 97071
mesh_service  |   Last vert scan: 97065, orig: 6
mesh_service  |   Last occupied scan: 14550, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 14551 active voxels with 97071 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 27744000
mesh_service  |   Active voxels with triangles: 14551
mesh_service  |   Total vertices needed: 97071
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 14551
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 14551
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=0, out_idx=0, global_voxel=22048, verts=3
mesh_service  | [EXTRACT] idx=2, out_idx=1, global_voxel=22070, verts=3
mesh_service  | [EXTRACT] idx=4, out_idx=2, global_voxel=22072, verts=6
mesh_service  | [EXTRACT] idx=5, out_idx=3, global_voxel=22073, verts=6
mesh_service  | [EXTRACT] idx=6, out_idx=4, global_voxel=22079, verts=6
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 14551
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 14551
mesh_service  |   Output size: 14551
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=3, scan=0
mesh_service  |   [1] verts=3, scan=3
mesh_service  |   [2] verts=6, scan=6
mesh_service  |   [3] verts=6, scan=12
mesh_service  |   [4] verts=6, scan=18
mesh_service  |   [5] verts=6, scan=24
mesh_service  |   [6] verts=3, scan=30
mesh_service  |   [7] verts=12, scan=33
mesh_service  |   [8] verts=9, scan=45
mesh_service  |   [9] verts=12, scan=54
mesh_service  |   Last element: verts=6, scan=97065, total=97071
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x767416000000
mesh_service  |   d_compressed_global_voxels ptr: 0x767442a27600
mesh_service  |   d_compressed_verts_scan ptr: 0x767442a43e00
mesh_service  |   d_vertex_buffer: 0x7673ea000000
mesh_service  |   d_normal_buffer: 0x7673e4000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 33x42x83
mesh_service  |   origin: [-0.9153, -2.1521, 0.5010]
mesh_service  |   num_active_voxels: 14551
mesh_service  |   voxel_size: 0.0500
mesh_service  |   iso_value: 0.0000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 228, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=22048, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=22070, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: voxel_idx=22072, vertex_offset=6
mesh_service  | [KERNEL] Thread 0: coords=[4,38,15], dims=[33,42,83]
mesh_service  | [KERNEL] Thread 1: coords=[26,38,15], dims=[33,42,83]
mesh_service  | [KERNEL] Thread 2: coords=[28,38,15], dims=[33,42,83]
mesh_service  | [KERNEL] Thread 0: base_idx=22048, max_idx=115038
mesh_service  | [KERNEL] Thread 1: base_idx=22070, max_idx=115038
mesh_service  | [KERNEL] Thread 2: base_idx=22072, max_idx=115038
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=8
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=1
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=96
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=6
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=3
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=6
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 0 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 2 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 11 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 1 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 1 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 2 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 11 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 61 µs
mesh_service  | Mesh generation completed in 14ms: 97071 vertices, 32357 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 14 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~11 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47176
mesh_service  |     - Points/sec: 3369714.2857
mesh_service  |     - Vertices generated: 97071
mesh_service  |     - Faces generated: 32357
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 97071 vertices, 32357 faces in 15ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 291213
mesh_service  | [DEBUG] Extracting colors for 47176 points
mesh_service  | [DEBUG] Color extraction complete, size: 141528
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 97071 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [-0.7153, -0.2124, 1.2510]
mesh_service  | [DEBUG Rerun] Vertex 1: [-0.7153, -0.2021, 1.2628]
mesh_service  | [DEBUG Rerun] Vertex 2: [-0.7060, -0.2021, 1.2510]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 0 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 37 µs
mesh_service  | [TIMING] Total Rerun publishing: 0 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 38 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 28 - Total Time: 15 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 15 ms (100.0000%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~12.0000 ms (estimate)
mesh_service  |      - Other: ~3.0000 ms (estimate)
mesh_service  |   2. Rerun Publishing: 0 ms (0.0000%)
mesh_service  |   3. Cleanup: 0 ms (0.0000%)
mesh_service  |   4. Other (metrics, etc): 0 ms
mesh_service  | Performance: 66.6667 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 15 ms
mesh_service  | [TIMING] Total message handling: 15 ms
mesh_service  | Received message, size: 320 bytes
mesh_service  | [TIMING] msgpack unpacking: 7 µs
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
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_29, keyframe_id: 29, type: keyframe.new, point_count: 47272
mesh_service  | [TIMING] Message parsing: 4 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 29 via RabbitMQ, shm_key: /slam3r_keyframe_29
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_29
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_29
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 27 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 7 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 47272, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Full pose matrix from shared memory (row-major):
mesh_service  |   [    0.9999,     0.0068,    -0.0128,     0.0000]
mesh_service  |   [   -0.0067,     1.0000,     0.0071,     0.0000]
mesh_service  |   [    0.0128,    -0.0070,     0.9999,     0.0000]
mesh_service  |   [   -0.0167,     0.0227,    -0.0016,     1.0000]
mesh_service  | [SHM DEBUG] Camera position (translation) from pose: [-0.0167, 0.0227, -0.0016]
mesh_service  | [SHM DEBUG] Checking alternate indices [3,7,11] (column-major): [0.0000, 0.0000, 0.0000]
mesh_service  | [SHM DEBUG] Full pose matrix as linear array:
mesh_service  |   [0-3]: 0.9999, 0.0068, -0.0128, 0.0000
mesh_service  |   [4-7]: -0.0067, 1.0000, 0.0071, 0.0000
mesh_service  |   [8-11]: 0.0128, -0.0070, 0.9999, 0.0000
mesh_service  |   [12-15]: -0.0167, 0.0227, -0.0016, 1.0000
mesh_service  | [SHM DEBUG] Header bbox: [-0.7211, -2.0876, 0.6965] to [0.5963, 1.0175, 7.8254]
mesh_service  | [SHM DEBUG] Calculated total size: 709184
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x76752852f000
mesh_service  | [TIMING] mmap full: 2 µs
mesh_service  | [TIMING] Total SharedMemory open: 78 µs (0.0780 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 47272 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [MESH GEN DEBUG] Received pose matrix from keyframe:
mesh_service  |   [    0.9999,     0.0068,    -0.0128,     0.0000]
mesh_service  |   [   -0.0067,     1.0000,     0.0071,     0.0000]
mesh_service  |   [    0.0128,    -0.0070,     0.9999,     0.0000]
mesh_service  |   [   -0.0167,     0.0227,    -0.0016,     1.0000]
mesh_service  | [MESH GEN DEBUG] Camera position from pose: [-0.0167, 0.0227, -0.0016]
mesh_service  | [TIMING] Spatial deduplication check: 1205 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.0586 m/s, points=47272
mesh_service  | [TIMING] Point/color data access: 0 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 47272 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [-0.7102, -0.6948, 1.0463]
mesh_service  |   Point 1: [-0.7087, -0.7004, 1.0536]
mesh_service  |   Point 2: [-0.7079, -0.7072, 1.0617]
mesh_service  |   Point 3: [-0.7081, -0.7134, 1.0706]
mesh_service  |   Point 4: [-0.7076, -0.7197, 1.0795]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [-0.7211, -2.0876, 0.6965]
mesh_service  |   Max: [0.5963, 1.0175, 7.8254]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [-0.7211, -2.0876, 0.6965]
mesh_service  |   Max: [0.5963, 1.0175, 7.8254]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 127 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.5410 MB)
mesh_service  | [TIMING] Octree update: 0 ms
mesh_service  | [CAMERA NORMAL PROVIDER] Computing normals for 47272 points with grid 185 x 1 x 1 and block 256
mesh_service  | [CAMERA NORMAL] Point 0: pos=[-0.710,-0.695,1.046], normal=[-0.479,-0.496,0.724], dist=1.447
mesh_service  | [CAMERA NORMAL] Point 1: pos=[-0.709,-0.700,1.054], normal=[-0.476,-0.497,0.726], dist=1.454
mesh_service  | [CAMERA NORMAL] Point 2: pos=[-0.708,-0.707,1.062], normal=[-0.472,-0.499,0.727], dist=1.463
mesh_service  | [CAMERA NORMAL] Point 3: pos=[-0.708,-0.713,1.071], normal=[-0.469,-0.500,0.728], dist=1.473
mesh_service  | [CAMERA NORMAL] Point 4: pos=[-0.708,-0.720,1.080], normal=[-0.466,-0.501,0.729], dist=1.482
mesh_service  | [CAMERA NORMAL] Point 5: pos=[-0.706,-0.726,1.089], normal=[-0.462,-0.502,0.731], dist=1.491
mesh_service  | [CAMERA NORMAL] Point 6: pos=[-0.705,-0.733,1.097], normal=[-0.459,-0.504,0.732], dist=1.501
mesh_service  | [CAMERA NORMAL] Point 7: pos=[-0.705,-0.740,1.107], normal=[-0.455,-0.505,0.733], dist=1.511
mesh_service  | [CAMERA NORMAL] Point 8: pos=[-0.704,-0.747,1.115], normal=[-0.452,-0.506,0.735], dist=1.521
mesh_service  | [CAMERA NORMAL] Point 9: pos=[-0.704,-0.754,1.126], normal=[-0.449,-0.507,0.736], dist=1.532
mesh_service  | [NORMAL ESTIMATION] Provider: 1 (Camera-based (fast))
mesh_service  | [NORMAL ESTIMATION] Time: 1 ms
mesh_service  | [NORMAL ESTIMATION] Points: 47272
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.0586 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 47272, Complexity: 0.4727
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 47272 points
mesh_service  | [MC DEBUG] Camera pose received (expecting row-major):
mesh_service  |   [0-3]: 0.9999, 0.0068, -0.0128, 0.0000
mesh_service  |   [4-7]: -0.0067, 1.0000, 0.0071, 0.0000
mesh_service  |   [8-11]: 0.0128, -0.0070, 0.9999, 0.0000
mesh_service  |   [12-15]: -0.0167, 0.0227, -0.0016, 1.0000
mesh_service  | [MC DEBUG] Camera position should be at [12,13,14]: [-0.0167, 0.0227, -0.0016]
mesh_service  | [MC BOUNDS UPDATE] Point cloud bounds changed, re-initializing TSDF
mesh_service  |   Old bounds: [-0.9153,-2.1521,0.5010] to [0.7347,-0.0521,4.6510]
mesh_service  |   New bounds: [-0.9158,-2.1519,0.4965] to [0.7366,-0.0701,4.7012]
mesh_service  | [SIMPLE TSDF INIT] Initializing TSDF volume:
mesh_service  |   Volume min: [-0.9158, -2.1519, 0.4965]
mesh_service  |   Volume max: [0.7366, -0.0701, 4.7012]
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Volume dimensions: [33, 42, 84]
mesh_service  |   Total voxels: 116424
mesh_service  | SimpleTSDF initialized:
mesh_service  |   Volume dims: 33x42x84
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Memory usage: 0.8882 MB
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Received camera pose pointer: 0x76752852f010
mesh_service  | [TSDF DEBUG] Interpreting pose matrix as ROW-MAJOR:
mesh_service  |   [    0.9999,     0.0068,    -0.0128,     0.0000]
mesh_service  |   [   -0.0067,     1.0000,     0.0071,     0.0000]
mesh_service  |   [    0.0128,    -0.0070,     0.9999,     0.0000]
mesh_service  |   [   -0.0167,     0.0227,    -0.0016,     1.0000]
mesh_service  | [TSDF DEBUG] Translation option 1 (indices 3,7,11): [0.0000, 0.0000, 0.0000]
mesh_service  | [TSDF DEBUG] Translation option 2 (indices 12,13,14): [-0.0167, 0.0227, -0.0016]
mesh_service  | [TSDF DEBUG] Using option 2 (row-major) as camera position: [-0.0167, 0.0227, -0.0016]
mesh_service  | [TSDF DEBUG] Rotation matrix determinant (should be ~1): 1.0000
mesh_service  | [TSDF DEBUG] integrate() called with 47272 points
mesh_service  | [TSDF DEBUG] Normal provider: External normals provided
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.1000 m
mesh_service  | [TIMING] TSDF debug copy: 6 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [-0.7102, -0.6948, 1.0463]
mesh_service  |   Point 1: [-0.7087, -0.7004, 1.0536]
mesh_service  |   Point 2: [-0.7079, -0.7072, 1.0617]
mesh_service  |   Point 3: [-0.7081, -0.7134, 1.0706]
mesh_service  |   Point 4: [-0.7076, -0.7197, 1.0795]
mesh_service  |   Point 5: [-0.7062, -0.7256, 1.0888]
mesh_service  |   Point 6: [-0.7050, -0.7330, 1.0971]
mesh_service  |   Point 7: [-0.7046, -0.7400, 1.1065]
mesh_service  |   Point 8: [-0.7039, -0.7467, 1.1154]
mesh_service  |   Point 9: [-0.7037, -0.7542, 1.1255]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.0000, Translation X: 18.3165
mesh_service  |   Scale Y: 20.0000, Translation Y: 43.0384
mesh_service  |   Scale Z: 20.0000, Translation Z: -9.9308
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 33x42x84
mesh_service  |   Origin: [-0.9158, -2.1519, 0.4965]
mesh_service  |   Voxel size: 0.0500
mesh_service  |   Truncation distance: 0.1000
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.710,-0.695,1.046] -> voxel=[4.113,29.143,10.995], normal=[-0.479,-0.496,0.724]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.709,-0.700,1.054] -> voxel=[4.143,29.030,11.141], normal=[-0.476,-0.497,0.726]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.708,-0.707,1.062] -> voxel=[4.159,28.894,11.304], normal=[-0.472,-0.499,0.727]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.708,-0.713,1.071] -> voxel=[4.155,28.771,11.481], normal=[-0.469,-0.500,0.728]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.708,-0.720,1.080] -> voxel=[4.164,28.644,11.660], normal=[-0.466,-0.501,0.729]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[2,27,8], max=[7,32,13], volume_dims=[33,42,84]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[2,27,9], max=[7,32,14], volume_dims=[33,42,84]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,84]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,84]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[2,26,9], max=[7,31,14], volume_dims=[33,42,84]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 116424
mesh_service  |   Modified voxels: 116244
mesh_service  |   Weighted voxels: 20231
mesh_service  | [MC DEBUG] TSDF sign distribution (weighted voxels only):
mesh_service  |   Positive (empty): 0
mesh_service  |   Negative (occupied): 0
mesh_service  |   Near zero: 0
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 33x42x84
mesh_service  |   Origin: [-0.9158, -2.1519, 0.4965]
mesh_service  |   Voxel size: 0.0500
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 40462 active voxels (safety margin from 20231 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 20231 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 6791, weight=17.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 6792, weight=22.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 6793, weight=14.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 6794, weight=8.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 26240, weight=10.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 26261, weight=2.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 26262, weight=19.000, pos=6
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 26263, weight=15.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 26264, weight=10.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 26270, weight=5.000, pos=9
mesh_service  | [MC DEBUG] Found 20231 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 
mesh_service  | [MC DEBUG] Classifying 20231 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 6791): coords=[26,37,4]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 6792): coords=[27,37,4]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 6793): coords=[28,37,4]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 6794): coords=[29,37,4]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 26240): coords=[5,39,18]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.081,0.064,0.072,0.088,0.049,0.024,0.065,0.087]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.064,0.014,0.067,0.072,0.024,0.018,0.044,0.065]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [0.014,-0.099,-0.098,0.067,0.018,-0.045,-0.094,0.044]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [-0.099,1.000,1.000,-0.098,-0.045,1.000,1.000,-0.094]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [0.065,1.000,1.000,0.053,0.090,1.000,1.000,0.037]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=102, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=153, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=0, num_verts=0
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 0 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 20231
mesh_service  |   Active voxels with triangles: 14847
mesh_service  |   Total vertices to generate: 99015
mesh_service  |   Last vert scan: 99009, orig: 6
mesh_service  |   Last occupied scan: 14846, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 14847 active voxels with 99015 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 27744000
mesh_service  |   Active voxels with triangles: 14847
mesh_service  |   Total vertices needed: 99015
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 14847
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 14847
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=2, out_idx=0, global_voxel=6793, verts=6
mesh_service  | [EXTRACT] idx=3, out_idx=1, global_voxel=6794, verts=6
mesh_service  | [EXTRACT] idx=6, out_idx=2, global_voxel=26262, verts=3
mesh_service  | [EXTRACT] idx=7, out_idx=3, global_voxel=26263, verts=9
mesh_service  | [EXTRACT] idx=8, out_idx=4, global_voxel=26264, verts=6
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 14847
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 14847
mesh_service  |   Output size: 14847
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=6, scan=0
mesh_service  |   [1] verts=6, scan=6
mesh_service  |   [2] verts=3, scan=12
mesh_service  |   [3] verts=9, scan=15
mesh_service  |   [4] verts=6, scan=24
mesh_service  |   [5] verts=12, scan=30
mesh_service  |   [6] verts=6, scan=42
mesh_service  |   [7] verts=3, scan=48
mesh_service  |   [8] verts=12, scan=51
mesh_service  |   [9] verts=9, scan=63
mesh_service  |   Last element: verts=6, scan=99009, total=99015
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x767416000000
mesh_service  |   d_compressed_global_voxels ptr: 0x767442a28200
mesh_service  |   d_compressed_verts_scan ptr: 0x767442a45200
mesh_service  |   d_vertex_buffer: 0x7673ea000000
mesh_service  |   d_normal_buffer: 0x7673e4000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 33x42x84
mesh_service  |   origin: [-0.9158, -2.1519, 0.4965]
mesh_service  |   num_active_voxels: 14847
mesh_service  |   voxel_size: 0.0500
mesh_service  |   iso_value: 0.0000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 232, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=6793, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=6794, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: voxel_idx=26262, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: coords=[28,37,4], dims=[33,42,84]
mesh_service  | [KERNEL] Thread 1: coords=[29,37,4], dims=[33,42,84]
mesh_service  | [KERNEL] Thread 2: coords=[27,39,18], dims=[33,42,84]
mesh_service  | [KERNEL] Thread 0: base_idx=6793, max_idx=116424
mesh_service  | [KERNEL] Thread 1: base_idx=6794, max_idx=116424
mesh_service  | [KERNEL] Thread 2: base_idx=26262, max_idx=116424
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=102
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=153
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=2
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
mesh_service  | [TIMING] Output copy (D2H): 0 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 1 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 0 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 0 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 2 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 66 µs
mesh_service  | Mesh generation completed in 5ms: 99015 vertices, 33005 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 5 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~2 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47272
mesh_service  |     - Points/sec: 9454400.0000
mesh_service  |     - Vertices generated: 99015
mesh_service  |     - Faces generated: 33005
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 99015 vertices, 33005 faces in 5ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 297045
mesh_service  | [DEBUG] Extracting colors for 47272 points
mesh_service  | [DEBUG] Color extraction complete, size: 141816
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 99015 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [0.4905, -0.3019, 0.6965]
mesh_service  | [DEBUG Rerun] Vertex 1: [0.5045, -0.2519, 0.6965]
mesh_service  | [DEBUG Rerun] Vertex 2: [0.4983, -0.3019, 0.7465]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 0 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 35 µs
mesh_service  | [TIMING] Total Rerun publishing: 0 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 28 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 29 - Total Time: 6 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 5 ms (83.3333%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~4.0000 ms (estimate)
mesh_service  |      - Other: ~1.0000 ms (estimate)
mesh_service  |   2. Rerun Publishing: 0 ms (0.0000%)
mesh_service  |   3. Cleanup: 0 ms (0.0000%)
mesh_service  |   4. Other (metrics, etc): 1 ms
mesh_service  | Performance: 166.6667 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 6 ms
mesh_service  | [TIMING] Total message handling: 6 ms

<LOG>

<LOG SLAM3R>
slam3r  | 2025-07-17 13:58:40,647 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:40,647 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 9.9994302e-01 -5.8880965e-03  8.9005521e-03 -1.0826178e-02]
slam3r  |  [ 5.8805193e-03  9.9998236e-01  8.7665493e-04  9.0545602e-03]
slam3r  |  [-8.9055495e-03 -8.2427129e-04  9.9995995e-01  1.4790893e-02]
slam3r  |  [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
slam3r  | 2025-07-17 13:58:40,647 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:40,649 - __main__ - INFO - Publishing keyframe 21: 47230 valid points (from 50176 total, 47230 passed confidence)
slam3r  | 2025-07-17 13:58:40,649 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:40,649 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:40,649 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:40,649 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0059,     0.0089,    -0.0108]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0059,     1.0000,     0.0009,     0.0091]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0089,    -0.0008,     1.0000,     0.0148]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0108, 0.0091, 0.0148]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0108, 0.0091, 0.0148]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0108, 0.0091, 0.0148]
slam3r  | 2025-07-17 13:58:40,650 - shared_memory - INFO - Wrote keyframe 21 to shared memory: /slam3r_keyframe_21
slam3r  | 2025-07-17 13:58:40,651 - shared_memory - INFO - Keyframe 21 bbox: min=(-0.71, -2.05, 0.69), max=(0.57, 1.01, 7.85)
slam3r  | 2025-07-17 13:58:40,652 - streaming_slam3r - INFO - Processing frame 101 with timestamp 6862345427
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.56it/s]
slam3r  | 2025-07-17 13:58:40,662 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:40,681 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,681 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.812]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:40,706 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,706 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.001, max: 210.659
slam3r  | 2025-07-17 13:58:40,730 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,730 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.825]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:40,756 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,756 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.001, max: 210.166
slam3r  | 2025-07-17 13:58:40,779 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,779 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.977]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:40,805 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,805 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.001, max: 210.448
slam3r  | 2025-07-17 13:58:40,827 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,827 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.893]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:40,853 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,853 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 211.676
slam3r  | 2025-07-17 13:58:40,856 - shared_memory - INFO - Published keyframe 21 to mesh service (47230 points)
slam3r  | 2025-07-17 13:58:40,856 - __main__ - INFO - Successfully published keyframe 21 with 47230 points
slam3r  | 2025-07-17 13:58:40,857 - streaming_slam3r - INFO - Processing frame 102 with timestamp 6930289441
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.28it/s]
slam3r  | 2025-07-17 13:58:40,868 - streaming_slam3r - INFO - Processing frame 103 with timestamp 6998233455
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.09it/s]
slam3r  | 2025-07-17 13:58:40,879 - streaming_slam3r - INFO - Processing frame 104 with timestamp 7066177469
encoding images: 100%|██████████| 1/1 [00:00<00:00, 113.79it/s]
slam3r  | 2025-07-17 13:58:40,931 - streaming_slam3r - INFO - Processing frame 105 with timestamp 7134121483
encoding images: 100%|██████████| 1/1 [00:00<00:00, 67.65it/s]
slam3r  | 2025-07-17 13:58:40,948 - streaming_slam3r - INFO - Frame 105 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:40,965 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,965 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.031]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:40,987 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:40,987 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 200.220
slam3r  | 2025-07-17 13:58:40,990 - streaming_slam3r - INFO - Returning keyframe result for frame 105
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - Publishing keyframe 22 with frame_id 105
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.015, 0.008, 0.012]
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:40,990 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 0.9999116  -0.00597199  0.01187974 -0.01468369]
slam3r  |  [ 0.00594885  0.99998033  0.00198246  0.00768905]
slam3r  |  [-0.01189135 -0.00191162  0.99992746  0.01185966]
slam3r  |  [ 0.          0.          0.          1.        ]]
slam3r  | 2025-07-17 13:58:40,991 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:40,993 - __main__ - INFO - Publishing keyframe 22: 47222 valid points (from 50176 total, 47222 passed confidence)
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0060,     0.0119,    -0.0147]
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0059,     1.0000,     0.0020,     0.0077]
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0119,    -0.0019,     0.9999,     0.0119]
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:40,993 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0147, 0.0077, 0.0119]
slam3r  | 2025-07-17 13:58:40,994 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0147, 0.0077, 0.0119]
slam3r  | 2025-07-17 13:58:40,994 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0147, 0.0077, 0.0119]
slam3r  | 2025-07-17 13:58:40,995 - shared_memory - INFO - Wrote keyframe 22 to shared memory: /slam3r_keyframe_22
slam3r  | 2025-07-17 13:58:40,995 - shared_memory - INFO - Keyframe 22 bbox: min=(-0.72, -2.06, 0.70), max=(0.57, 1.01, 7.76)
slam3r  | 2025-07-17 13:58:40,996 - streaming_slam3r - INFO - Processing frame 106 with timestamp 7202065498
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.73it/s]
slam3r  | 2025-07-17 13:58:41,006 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:41,026 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,026 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.720]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,051 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,051 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 203.782
slam3r  | 2025-07-17 13:58:41,078 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,078 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.155]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,103 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,103 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 193.623
slam3r  | 2025-07-17 13:58:41,124 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,124 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.163]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,150 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,150 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 212.537
slam3r  | 2025-07-17 13:58:41,172 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,172 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.400]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,197 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,197 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 204.945
slam3r  | 2025-07-17 13:58:41,200 - shared_memory - INFO - Published keyframe 22 to mesh service (47222 points)
slam3r  | 2025-07-17 13:58:41,200 - __main__ - INFO - Successfully published keyframe 22 with 47222 points
slam3r  | 2025-07-17 13:58:41,201 - streaming_slam3r - INFO - Processing frame 107 with timestamp 7270009512
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.19it/s]
slam3r  | 2025-07-17 13:58:41,212 - streaming_slam3r - INFO - Processing frame 108 with timestamp 7337953526
encoding images: 100%|██████████| 1/1 [00:00<00:00, 113.11it/s]
slam3r  | 2025-07-17 13:58:41,222 - streaming_slam3r - INFO - Processing frame 109 with timestamp 7405897540
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.06it/s]
slam3r  | 2025-07-17 13:58:41,267 - streaming_slam3r - INFO - Processing frame 110 with timestamp 7473841554
encoding images: 100%|██████████| 1/1 [00:00<00:00, 74.84it/s]
slam3r  | 2025-07-17 13:58:41,282 - streaming_slam3r - INFO - Frame 110 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:41,300 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,301 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.719]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,323 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,323 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 193.505
slam3r  | 2025-07-17 13:58:41,325 - streaming_slam3r - INFO - Returning keyframe result for frame 110
slam3r  | 2025-07-17 13:58:41,325 - __main__ - INFO - Publishing keyframe 23 with frame_id 110
slam3r  | 2025-07-17 13:58:41,325 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:41,325 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:41,325 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:41,325 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:41,326 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:41,326 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:41,326 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.015, 0.005, 0.016]
slam3r  | 2025-07-17 13:58:41,326 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:41,326 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 0.99989414 -0.00647535  0.01302702 -0.01527504]
slam3r  |  [ 0.00641918  0.99996996  0.00434846  0.00534571]
slam3r  |  [-0.01305479 -0.00426435  0.99990565  0.01572883]
slam3r  |  [ 0.          0.          0.          1.        ]]
slam3r  | 2025-07-17 13:58:41,326 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:41,328 - __main__ - INFO - Publishing keyframe 23: 47197 valid points (from 50176 total, 47197 passed confidence)
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0065,     0.0130,    -0.0153]
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0064,     1.0000,     0.0043,     0.0053]
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0131,    -0.0043,     0.9999,     0.0157]
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0153, 0.0053, 0.0157]
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0153, 0.0053, 0.0157]
slam3r  | 2025-07-17 13:58:41,329 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0153, 0.0053, 0.0157]
slam3r  | 2025-07-17 13:58:41,330 - shared_memory - INFO - Wrote keyframe 23 to shared memory: /slam3r_keyframe_23
slam3r  | 2025-07-17 13:58:41,330 - shared_memory - INFO - Keyframe 23 bbox: min=(-0.72, -2.07, 0.71), max=(0.58, 1.01, 7.72)
slam3r  | 2025-07-17 13:58:41,331 - shared_memory - INFO - Published keyframe 23 to mesh service (47197 points)
slam3r  | 2025-07-17 13:58:41,331 - __main__ - INFO - Successfully published keyframe 23 with 47197 points
slam3r  | 2025-07-17 13:58:41,332 - streaming_slam3r - INFO - Processing frame 111 with timestamp 7541785568
encoding images: 100%|██████████| 1/1 [00:00<00:00, 101.98it/s]
slam3r  | 2025-07-17 13:58:41,343 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:41,362 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,362 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.049]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,387 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,387 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 211.160
slam3r  | 2025-07-17 13:58:41,413 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,413 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.050]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,439 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,439 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 219.152
slam3r  | 2025-07-17 13:58:41,463 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,463 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.335]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,489 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,489 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 218.586
slam3r  | 2025-07-17 13:58:41,510 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,510 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.141]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,535 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,536 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 215.408
slam3r  | 2025-07-17 13:58:41,540 - streaming_slam3r - INFO - Processing frame 112 with timestamp 7609729582
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.71it/s]
slam3r  | 2025-07-17 13:58:41,551 - streaming_slam3r - INFO - Processing frame 113 with timestamp 7677673596
encoding images: 100%|██████████| 1/1 [00:00<00:00, 113.83it/s]
slam3r  | 2025-07-17 13:58:41,561 - streaming_slam3r - INFO - Processing frame 114 with timestamp 7745617611
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.00it/s]
slam3r  | 2025-07-17 13:58:41,604 - streaming_slam3r - INFO - Processing frame 115 with timestamp 7813561625
encoding images: 100%|██████████| 1/1 [00:00<00:00, 105.91it/s]
slam3r  | 2025-07-17 13:58:41,615 - streaming_slam3r - INFO - Frame 115 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:41,634 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,634 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.005]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,657 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,657 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 206.677
slam3r  | 2025-07-17 13:58:41,659 - streaming_slam3r - INFO - Returning keyframe result for frame 115
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - Publishing keyframe 24 with frame_id 115
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:41,659 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:41,660 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.011, 0.011, -0.003]
slam3r  | 2025-07-17 13:58:41,660 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:41,660 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 9.9993509e-01 -6.3388562e-03  9.4690779e-03 -1.0751814e-02]
slam3r  |  [ 6.3324827e-03  9.9997973e-01  7.0304575e-04  1.1177026e-02]
slam3r  |  [-9.4733266e-03 -6.4303080e-04  9.9995494e-01 -3.2008886e-03]
slam3r  |  [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
slam3r  | 2025-07-17 13:58:41,660 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:41,662 - __main__ - INFO - Publishing keyframe 24: 47237 valid points (from 50176 total, 47237 passed confidence)
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0063,     0.0095,    -0.0108]
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0063,     1.0000,     0.0007,     0.0112]
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0095,    -0.0006,     1.0000,    -0.0032]
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0108, 0.0112, -0.0032]
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0108, 0.0112, -0.0032]
slam3r  | 2025-07-17 13:58:41,662 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0108, 0.0112, -0.0032]
slam3r  | 2025-07-17 13:58:41,663 - shared_memory - INFO - Wrote keyframe 24 to shared memory: /slam3r_keyframe_24
slam3r  | 2025-07-17 13:58:41,663 - shared_memory - INFO - Keyframe 24 bbox: min=(-0.72, -2.07, 0.70), max=(0.58, 1.01, 7.79)
slam3r  | 2025-07-17 13:58:41,664 - shared_memory - INFO - Published keyframe 24 to mesh service (47237 points)
slam3r  | 2025-07-17 13:58:41,664 - __main__ - INFO - Successfully published keyframe 24 with 47237 points
slam3r  | 2025-07-17 13:58:41,672 - streaming_slam3r - INFO - Processing frame 116 with timestamp 7881505639
encoding images: 100%|██████████| 1/1 [00:00<00:00, 102.41it/s]
slam3r  | 2025-07-17 13:58:41,683 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:41,701 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,701 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.748]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,726 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,727 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 212.653
slam3r  | 2025-07-17 13:58:41,750 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,750 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.954]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,775 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,775 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 212.529
slam3r  | 2025-07-17 13:58:41,797 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,797 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.921]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,822 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,822 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 214.955
slam3r  | 2025-07-17 13:58:41,844 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,844 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.894]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,870 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,870 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 214.569
slam3r  | 2025-07-17 13:58:41,874 - streaming_slam3r - INFO - Processing frame 117 with timestamp 7949449653
encoding images: 100%|██████████| 1/1 [00:00<00:00, 103.33it/s]
slam3r  | 2025-07-17 13:58:41,886 - streaming_slam3r - INFO - Processing frame 118 with timestamp 8017393667
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.90it/s]
slam3r  | 2025-07-17 13:58:41,897 - streaming_slam3r - INFO - Processing frame 119 with timestamp 8085337681
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.02it/s]
slam3r  | 2025-07-17 13:58:41,907 - __main__ - INFO - FPS: 14.72 | Frames: 120 | Keyframes: 24 | Segment frames: 120
slam3r  | 2025-07-17 13:58:41,944 - streaming_slam3r - INFO - Processing frame 120 with timestamp 8153281695
encoding images: 100%|██████████| 1/1 [00:00<00:00, 101.16it/s]
slam3r  | 2025-07-17 13:58:41,955 - streaming_slam3r - INFO - Frame 120 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:41,975 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,975 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.995]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:41,997 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:41,998 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 202.214
slam3r  | 2025-07-17 13:58:42,000 - streaming_slam3r - INFO - Returning keyframe result for frame 120
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - Publishing keyframe 25 with frame_id 120
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.009, 0.024, -0.030]
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 0.9999574  -0.00563776  0.00731564 -0.00906066]
slam3r  |  [ 0.0056861   0.99996203 -0.00660299  0.02402437]
slam3r  |  [-0.00727814  0.0066443   0.9999514  -0.02991068]
slam3r  |  [ 0.          0.          0.          1.        ]]
slam3r  | 2025-07-17 13:58:42,001 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:42,003 - __main__ - INFO - Publishing keyframe 25: 47162 valid points (from 50176 total, 47162 passed confidence)
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,    -0.0056,     0.0073,    -0.0091]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0057,     1.0000,    -0.0066,     0.0240]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0073,     0.0066,     1.0000,    -0.0299]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0091, 0.0240, -0.0299]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0091, 0.0240, -0.0299]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0091, 0.0240, -0.0299]
slam3r  | 2025-07-17 13:58:42,004 - shared_memory - INFO - Wrote keyframe 25 to shared memory: /slam3r_keyframe_25
slam3r  | 2025-07-17 13:58:42,005 - shared_memory - INFO - Keyframe 25 bbox: min=(-0.72, -2.07, 0.69), max=(0.57, 1.01, 7.83)
slam3r  | 2025-07-17 13:58:42,005 - shared_memory - INFO - Published keyframe 25 to mesh service (47162 points)
slam3r  | 2025-07-17 13:58:42,005 - __main__ - INFO - Successfully published keyframe 25 with 47162 points
slam3r  | 2025-07-17 13:58:42,012 - streaming_slam3r - INFO - Processing frame 121 with timestamp 8221225710
encoding images: 100%|██████████| 1/1 [00:00<00:00, 105.98it/s]
slam3r  | 2025-07-17 13:58:42,023 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:42,042 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,042 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.958]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,068 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,068 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 206.608
slam3r  | 2025-07-17 13:58:42,093 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,093 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.420]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,119 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,119 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 208.234
slam3r  | 2025-07-17 13:58:42,141 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,141 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.220]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,166 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,166 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 209.753
slam3r  | 2025-07-17 13:58:42,187 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,187 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.122]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,212 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,212 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 209.220
slam3r  | 2025-07-17 13:58:42,217 - streaming_slam3r - INFO - Processing frame 122 with timestamp 8289169724
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.00it/s]
slam3r  | 2025-07-17 13:58:42,228 - streaming_slam3r - INFO - Processing frame 123 with timestamp 8357113738
encoding images: 100%|██████████| 1/1 [00:00<00:00, 110.73it/s]
slam3r  | 2025-07-17 13:58:42,238 - streaming_slam3r - INFO - Processing frame 124 with timestamp 8425057752
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.93it/s]
slam3r  | 2025-07-17 13:58:42,285 - streaming_slam3r - INFO - Processing frame 125 with timestamp 8493001766
encoding images: 100%|██████████| 1/1 [00:00<00:00, 82.52it/s]
slam3r  | 2025-07-17 13:58:42,298 - streaming_slam3r - INFO - Frame 125 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:42,317 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,317 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.391]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,339 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,339 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.001, max: 202.866
slam3r  | 2025-07-17 13:58:42,342 - streaming_slam3r - INFO - Returning keyframe result for frame 125
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - Publishing keyframe 26 with frame_id 125
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.012, 0.022, -0.029]
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:42,342 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 0.9999397  -0.00621032  0.00905677 -0.0116978 ]
slam3r  |  [ 0.00625804  0.9999666  -0.00525119  0.02238337]
slam3r  |  [-0.00902387  0.0053075   0.9999452  -0.02864587]
slam3r  |  [ 0.          0.          0.          1.        ]]
slam3r  | 2025-07-17 13:58:42,343 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:42,345 - __main__ - INFO - Publishing keyframe 26: 47197 valid points (from 50176 total, 47197 passed confidence)
slam3r  | 2025-07-17 13:58:42,345 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:42,345 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:42,345 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0062,     0.0091,    -0.0117]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0063,     1.0000,    -0.0053,     0.0224]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0090,     0.0053,     0.9999,    -0.0286]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0117, 0.0224, -0.0286]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0117, 0.0224, -0.0286]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0117, 0.0224, -0.0286]
slam3r  | 2025-07-17 13:58:42,346 - shared_memory - INFO - Wrote keyframe 26 to shared memory: /slam3r_keyframe_26
slam3r  | 2025-07-17 13:58:42,347 - shared_memory - INFO - Keyframe 26 bbox: min=(-0.72, -2.07, 0.69), max=(0.53, 1.01, 7.90)
slam3r  | 2025-07-17 13:58:42,347 - shared_memory - INFO - Published keyframe 26 to mesh service (47197 points)
slam3r  | 2025-07-17 13:58:42,347 - __main__ - INFO - Successfully published keyframe 26 with 47197 points
slam3r  | 2025-07-17 13:58:42,351 - streaming_slam3r - INFO - Processing frame 126 with timestamp 8560945780
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.60it/s]
slam3r  | 2025-07-17 13:58:42,361 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:42,380 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,380 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.326]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,406 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,406 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 208.647
slam3r  | 2025-07-17 13:58:42,430 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,430 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.237]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,455 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,455 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.001, max: 207.678
slam3r  | 2025-07-17 13:58:42,477 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,477 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.323]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,502 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,502 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.001, max: 207.481
slam3r  | 2025-07-17 13:58:42,525 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,525 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.556]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,550 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,550 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 210.843
slam3r  | 2025-07-17 13:58:42,554 - streaming_slam3r - INFO - Processing frame 127 with timestamp 8628889794
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.72it/s]
slam3r  | 2025-07-17 13:58:42,565 - streaming_slam3r - INFO - Processing frame 128 with timestamp 8696833808
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.92it/s]
slam3r  | 2025-07-17 13:58:42,576 - streaming_slam3r - INFO - Processing frame 129 with timestamp 8764777823
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.15it/s]
slam3r  | 2025-07-17 13:58:42,631 - streaming_slam3r - INFO - Processing frame 130 with timestamp 8832721837
encoding images: 100%|██████████| 1/1 [00:00<00:00, 64.89it/s]
slam3r  | 2025-07-17 13:58:42,648 - streaming_slam3r - INFO - Frame 130 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:42,666 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,666 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 28.162]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,688 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,688 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 204.942
slam3r  | 2025-07-17 13:58:42,691 - streaming_slam3r - INFO - Returning keyframe result for frame 130
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - Publishing keyframe 27 with frame_id 130
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.011, 0.020, -0.022]
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:42,691 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 0.99993986 -0.00635927  0.00893422 -0.01128408]
slam3r  |  [ 0.00640583  0.99996597 -0.00519342  0.0204387 ]
slam3r  |  [-0.0089009   0.00525031  0.9999466  -0.02248311]
slam3r  |  [ 0.          0.          0.          1.        ]]
slam3r  | 2025-07-17 13:58:42,692 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:42,694 - __main__ - INFO - Publishing keyframe 27: 47179 valid points (from 50176 total, 47179 passed confidence)
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0064,     0.0089,    -0.0113]
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0064,     1.0000,    -0.0052,     0.0204]
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0089,     0.0053,     0.9999,    -0.0225]
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:42,694 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0113, 0.0204, -0.0225]
slam3r  | 2025-07-17 13:58:42,695 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0113, 0.0204, -0.0225]
slam3r  | 2025-07-17 13:58:42,695 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0113, 0.0204, -0.0225]
slam3r  | 2025-07-17 13:58:42,695 - shared_memory - INFO - Wrote keyframe 27 to shared memory: /slam3r_keyframe_27
slam3r  | 2025-07-17 13:58:42,696 - shared_memory - INFO - Keyframe 27 bbox: min=(-0.72, -2.08, 0.69), max=(0.53, 1.01, 7.94)
slam3r  | 2025-07-17 13:58:42,697 - streaming_slam3r - INFO - Processing frame 131 with timestamp 8900665851
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.32it/s]
slam3r  | 2025-07-17 13:58:42,707 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:42,725 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,726 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.954]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,750 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,751 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 201.758
slam3r  | 2025-07-17 13:58:42,773 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,774 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.675]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,799 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,799 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 204.919
slam3r  | 2025-07-17 13:58:42,820 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,821 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 28.207]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,846 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,846 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 212.459
slam3r  | 2025-07-17 13:58:42,867 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,867 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.814]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:42,892 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:42,892 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 208.879
slam3r  | 2025-07-17 13:58:42,896 - shared_memory - INFO - Published keyframe 27 to mesh service (47179 points)
slam3r  | 2025-07-17 13:58:42,896 - __main__ - INFO - Successfully published keyframe 27 with 47179 points
slam3r  | 2025-07-17 13:58:42,897 - streaming_slam3r - INFO - Processing frame 132 with timestamp 8968609865
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.05it/s]
slam3r  | 2025-07-17 13:58:42,908 - streaming_slam3r - INFO - Processing frame 133 with timestamp 9036553879
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.58it/s]
slam3r  | 2025-07-17 13:58:42,918 - streaming_slam3r - INFO - Processing frame 134 with timestamp 9104497893
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.13it/s]
slam3r  | 2025-07-17 13:58:42,968 - streaming_slam3r - INFO - Processing frame 135 with timestamp 9172441907
encoding images: 100%|██████████| 1/1 [00:00<00:00, 68.28it/s]
slam3r  | 2025-07-17 13:58:42,985 - streaming_slam3r - INFO - Frame 135 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:43,003 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,003 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.270]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,025 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,025 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 183.714
slam3r  | 2025-07-17 13:58:43,027 - streaming_slam3r - INFO - Returning keyframe result for frame 135
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - Publishing keyframe 28 with frame_id 135
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.017, 0.011, 0.015]
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 9.99859631e-01 -7.17250304e-03  1.51386466e-02 -1.71692148e-02]
slam3r  |  [ 7.17148418e-03  9.99974251e-01  1.21403908e-04  1.12826638e-02]
slam3r  |  [-1.51391355e-02 -1.27995518e-05  9.99885440e-01  1.50980949e-02]
slam3r  |  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
slam3r  | 2025-07-17 13:58:43,028 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:43,030 - __main__ - INFO - Publishing keyframe 28: 47176 valid points (from 50176 total, 47176 passed confidence)
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0072,     0.0151,    -0.0172]
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0072,     1.0000,     0.0001,     0.0113]
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0151,    -0.0000,     0.9999,     0.0151]
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0172, 0.0113, 0.0151]
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0172, 0.0113, 0.0151]
slam3r  | 2025-07-17 13:58:43,031 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0172, 0.0113, 0.0151]
slam3r  | 2025-07-17 13:58:43,032 - shared_memory - INFO - Wrote keyframe 28 to shared memory: /slam3r_keyframe_28
slam3r  | 2025-07-17 13:58:43,032 - shared_memory - INFO - Keyframe 28 bbox: min=(-0.72, -2.09, 0.70), max=(0.56, 1.02, 7.79)
slam3r  | 2025-07-17 13:58:43,033 - streaming_slam3r - INFO - Processing frame 136 with timestamp 9240385922
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.73it/s]
slam3r  | 2025-07-17 13:58:43,043 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:43,063 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,063 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.515]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,087 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,088 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 207.084
slam3r  | 2025-07-17 13:58:43,111 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,111 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.789]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,136 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,136 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 216.169
slam3r  | 2025-07-17 13:58:43,158 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,158 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.242]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,183 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,183 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 210.808
slam3r  | 2025-07-17 13:58:43,204 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,204 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.502]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,229 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,229 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 216.205
slam3r  | 2025-07-17 13:58:43,232 - shared_memory - INFO - Published keyframe 28 to mesh service (47176 points)
slam3r  | 2025-07-17 13:58:43,232 - __main__ - INFO - Successfully published keyframe 28 with 47176 points
slam3r  | 2025-07-17 13:58:43,233 - streaming_slam3r - INFO - Processing frame 137 with timestamp 9308329936
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.11it/s]
slam3r  | 2025-07-17 13:58:43,245 - streaming_slam3r - INFO - Processing frame 138 with timestamp 9376273950
encoding images: 100%|██████████| 1/1 [00:00<00:00, 110.62it/s]
slam3r  | 2025-07-17 13:58:43,256 - streaming_slam3r - INFO - Processing frame 139 with timestamp 9444217964
encoding images: 100%|██████████| 1/1 [00:00<00:00, 112.34it/s]
slam3r  | 2025-07-17 13:58:43,310 - streaming_slam3r - INFO - Processing frame 140 with timestamp 9512161978
encoding images: 100%|██████████| 1/1 [00:00<00:00, 63.38it/s]
slam3r  | 2025-07-17 13:58:43,328 - streaming_slam3r - INFO - Frame 140 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 13:58:43,346 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,346 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.996]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,368 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,368 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 213.185
slam3r  | 2025-07-17 13:58:43,371 - streaming_slam3r - INFO - Returning keyframe result for frame 140
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - Publishing keyframe 29 with frame_id 140
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose translation: [-0.017, 0.023, -0.002]
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Is identity matrix: False
slam3r  | 2025-07-17 13:58:43,371 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value:
slam3r  | [[ 0.9998953  -0.00670991  0.01281733 -0.01665397]
slam3r  |  [ 0.00680037  0.9999522  -0.00702658  0.02274812]
slam3r  |  [-0.01276957  0.00711298  0.9998932  -0.00162959]
slam3r  |  [ 0.          0.          0.          1.        ]]
slam3r  | 2025-07-17 13:58:43,372 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 13:58:43,374 - __main__ - INFO - Publishing keyframe 29: 47272 valid points (from 50176 total, 47272 passed confidence)
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.9999,    -0.0067,     0.0128,    -0.0167]
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0068,     1.0000,    -0.0070,     0.0227]
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [   -0.0128,     0.0071,     0.9999,    -0.0016]
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [-0.0167, 0.0227, -0.0016]
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R FIX] Original pose translation at [0,3],[1,3],[2,3]: [-0.0167, 0.0227, -0.0016]
slam3r  | 2025-07-17 13:58:43,374 - shared_memory - INFO - [SHM SLAM3R FIX] Row-major flattened translation at [12],[13],[14]: [-0.0167, 0.0227, -0.0016]
slam3r  | 2025-07-17 13:58:43,375 - shared_memory - INFO - Wrote keyframe 29 to shared memory: /slam3r_keyframe_29
slam3r  | 2025-07-17 13:58:43,375 - shared_memory - INFO - Keyframe 29 bbox: min=(-0.72, -2.09, 0.70), max=(0.60, 1.02, 7.83)
slam3r  | 2025-07-17 13:58:43,377 - streaming_slam3r - INFO - Processing frame 141 with timestamp 9580105992
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.23it/s]
slam3r  | 2025-07-17 13:58:43,386 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 13:58:43,405 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,405 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.891]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,430 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,431 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 209.473
slam3r  | 2025-07-17 13:58:43,454 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,455 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.929]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,480 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,480 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 210.767
slam3r  | 2025-07-17 13:58:43,502 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,502 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 27.085]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,527 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,527 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 214.023
slam3r  | 2025-07-17 13:58:43,548 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,548 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.000, 26.734]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 13:58:43,573 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 13:58:43,573 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.000, max: 209.783
slam3r  | 2025-07-17 13:58:43,577 - shared_memory - INFO - Published keyframe 29 to mesh service (47272 points)
slam3r  | 2025-07-17 13:58:43,577 - __main__ - INFO - Successfully published keyframe 29 with 47272 points
slam3r  | 2025-07-17 13:58:43,578 - streaming_slam3r - INFO - Processing frame 142 with timestamp 9648050006
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.73it/s]
slam3r  | 2025-07-17 13:58:43,589 - streaming_slam3r - INFO - Processing frame 143 with timestamp 9715994020
encoding images: 100%|██████████| 1/1 [00:00<00:00, 114.89it/s]
slam3r  | 2025-07-17 13:58:43,599 - streaming_slam3r - INFO - Processing frame 144 with timestamp 9783938035
<LOG SLAM3R>