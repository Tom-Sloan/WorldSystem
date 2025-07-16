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
esh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe  via RabbitMQ, shm_key: 
mesh_service  | [DEBUG] Opening shared memory segment: 
mesh_service  | [SHM DEBUG] Opening shared memory: 
mesh_service  | [TIMING] Keyframe handler total: 0 ms
mesh_service  | [TIMING] Total message handling: 1 ms
mesh_service  | [SHM DEBUG] shm_open failed, errno: 22 (Invalid argument)
mesh_service  | Failed to open shared memory segment: 
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
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_124, keyframe_id: 124, type: keyframe.new, point_count: 47368
mesh_service  | [TIMING] Message parsing: 5 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 124 via RabbitMQ, shm_key: /slam3r_keyframe_124
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_124
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_124
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 10 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 8 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 47368, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Header bbox: [-0.754, -3.215, 5.450] to [2.430, 0.912, 20.740]
mesh_service  | [SHM DEBUG] Calculated total size: 710624
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x74b74454e000
mesh_service  | [TIMING] mmap full: 2 µs
mesh_service  | [TIMING] Total SharedMemory open: 56 µs (0.056 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 47368 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [TIMING] Spatial deduplication check: 130 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.000 m/s, points=47368
mesh_service  | [TIMING] Point/color data access: 0 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 47368 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [-0.621, -0.855, 5.915]
mesh_service  |   Point 1: [-0.617, -0.860, 5.921]
mesh_service  |   Point 2: [-0.615, -0.868, 5.926]
mesh_service  |   Point 3: [-0.613, -0.873, 5.938]
mesh_service  |   Point 4: [-0.613, -0.883, 5.945]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [-0.754, -3.215, 5.450]
mesh_service  |   Max: [2.430, 0.912, 20.740]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [-0.754, -3.215, 5.450]
mesh_service  |   Max: [2.430, 0.912, 20.740]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 143 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.542 MB)
mesh_service  | [TIMING] Octree update: 0 ms
mesh_service  | [TIMING] Normal estimation: SKIPPED (using TSDF fallback)
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.000 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 47368, Complexity: 0.474
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 47368 points
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Camera position: [0.000, 0.000, 0.000]
mesh_service  | [TSDF DEBUG] integrate() called with 47368 points
mesh_service  | [TIMING] TSDF debug copy: 10 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [-0.621, -0.855, 5.915]
mesh_service  |   Point 1: [-0.617, -0.860, 5.921]
mesh_service  |   Point 2: [-0.615, -0.868, 5.926]
mesh_service  |   Point 3: [-0.613, -0.873, 5.938]
mesh_service  |   Point 4: [-0.613, -0.883, 5.945]
mesh_service  |   Point 5: [-0.614, -0.891, 5.964]
mesh_service  |   Point 6: [-0.613, -0.900, 5.972]
mesh_service  |   Point 7: [-0.612, -0.908, 5.986]
mesh_service  |   Point 8: [-0.613, -0.918, 6.002]
mesh_service  |   Point 9: [-0.610, -0.927, 6.009]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.000, Translation X: 40.000
mesh_service  |   Scale Y: 20.000, Translation Y: 60.000
mesh_service  |   Scale Z: 20.000, Translation Z: -0.000
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  |   Truncation distance: 0.100
mesh_service  | [TSDF KERNEL] Point 0: world=[-0.621,-0.855,5.915] -> voxel=[27.574,42.899,118.299], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[-0.617,-0.860,5.921] -> voxel=[27.666,42.796,118.417], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[-0.615,-0.868,5.926] -> voxel=[27.699,42.639,118.525], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[-0.613,-0.873,5.938] -> voxel=[27.731,42.533,118.757], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[-0.613,-0.883,5.945] -> voxel=[27.732,42.341,118.904], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[25,40,116], max=[30,45,121], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[25,40,116], max=[30,45,121], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[25,40,116], max=[30,45,121], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[25,40,116], max=[30,45,121], volume_dims=[80,100,200]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[25,40,116], max=[30,45,121], volume_dims=[80,100,200]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TSDF ERROR] Kernel launch failed: invalid configuration argument
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1600000
mesh_service  |   Modified voxels: 1599286
mesh_service  |   Weighted voxels: 170333
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 80x100x200
mesh_service  |   Origin: [-2.000, -3.000, 0.000]
mesh_service  |   Voxel size: 0.050
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 340666 active voxels (safety margin from 170333 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 170333 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117408, weight=100.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117409, weight=100.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117410, weight=100.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117248, weight=100.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117249, weight=100.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117250, weight=100.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108528, weight=100.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108529, weight=100.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 108530, weight=100.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 117407, weight=100.000, pos=6
mesh_service  | [MC DEBUG] Found 170333 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
mesh_service  | [MC DEBUG] Classifying 170333 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 117248): coords=[48,65,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 117249): coords=[49,65,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 117250): coords=[50,65,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 117408): coords=[48,67,14]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 117409): coords=[49,67,14]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [-0.070,-0.067,-0.065,-0.069,-0.008,-0.008,-0.029,-0.021]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [-0.067,-0.070,-0.072,-0.065,-0.008,0.002,-0.019,-0.029]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [-0.070,1.000,1.000,-0.072,0.002,1.000,1.000,-0.019]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [-0.080,-0.072,-0.064,-0.067,-0.005,-0.021,0.001,0.010]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [-0.072,-0.073,-0.067,-0.064,-0.021,0.002,0.009,0.001]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=255, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=223, num_verts=3
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=137, num_verts=9
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=63, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=31, num_verts=9
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 1 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 170333
mesh_service  |   Active voxels with triangles: 123755
mesh_service  |   Total vertices to generate: 853440
mesh_service  |   Last vert scan: 853440, orig: 0
mesh_service  |   Last occupied scan: 123755, orig: 0
mesh_service  | [MC DEBUG] Generating triangles for 123755 active voxels with 853440 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 1600000
mesh_service  |   Active voxels with triangles: 123755
mesh_service  |   Total vertices needed: 853440
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 123755
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 123755
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=1, out_idx=0, global_voxel=117249, verts=3
mesh_service  | [EXTRACT] idx=2, out_idx=1, global_voxel=117250, verts=9
mesh_service  | [EXTRACT] idx=3, out_idx=2, global_voxel=117408, verts=6
mesh_service  | [EXTRACT] idx=4, out_idx=3, global_voxel=117409, verts=9
mesh_service  | [EXTRACT] idx=5, out_idx=4, global_voxel=117410, verts=6
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 123755
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 123755
mesh_service  |   Output size: 123755
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=3, scan=0
mesh_service  |   [1] verts=9, scan=3
mesh_service  |   [2] verts=6, scan=12
mesh_service  |   [3] verts=9, scan=18
mesh_service  |   [4] verts=6, scan=27
mesh_service  |   [5] verts=3, scan=33
mesh_service  |   [6] verts=6, scan=36
mesh_service  |   [7] verts=6, scan=42
mesh_service  |   [8] verts=9, scan=48
mesh_service  |   [9] verts=3, scan=57
mesh_service  |   Last element: verts=15, scan=853425, total=853440
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x74b65ec00000
mesh_service  |   d_compressed_global_voxels ptr: 0x74b733600000
mesh_service  |   d_compressed_verts_scan ptr: 0x74b7336f1c00
mesh_service  |   d_vertex_buffer: 0x74b734000000
mesh_service  |   d_normal_buffer: 0x74b72e000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 80x100x200
mesh_service  |   origin: [-2.000, -3.000, 0.000]
mesh_service  |   num_active_voxels: 123755
mesh_service  |   voxel_size: 0.050
mesh_service  |   iso_value: 0.000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 1934, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=117249, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=117250, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: voxel_idx=117408, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: coords=[49,65,14], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 1: coords=[50,65,14], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 2: coords=[48,67,14], dims=[80,100,200]
mesh_service  | [KERNEL] Thread 0: base_idx=117249, max_idx=1600000
mesh_service  | [KERNEL] Thread 1: base_idx=117250, max_idx=1600000
mesh_service  | [KERNEL] Thread 2: base_idx=117408, max_idx=1600000
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=223
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=137
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=63
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=3
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=12
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 0 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 6 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 14 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 1 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 6 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 15 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 84 µs
mesh_service  | Mesh generation completed in 16ms: 853440 vertices, 284480 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 16 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal estimation: SKIPPED (0 ms)
mesh_service  |     - TSDF + Marching Cubes: ~15 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 47368
mesh_service  |     - Points/sec: 2960500.000
mesh_service  |     - Vertices generated: 853440
mesh_service  |     - Faces generated: 284480
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 853440 vertices, 284480 faces in 17ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 2560320
mesh_service  | [DEBUG] Extracting colors for 47368 points
mesh_service  | [DEBUG] Color extraction complete, size: 142104
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 853440 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [0.500, 0.250, 0.748]
mesh_service  | [DEBUG Rerun] Vertex 1: [0.489, 0.250, 0.750]
mesh_service  | [DEBUG Rerun] Vertex 2: [0.500, 0.256, 0.750]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 4 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 52 µs
mesh_service  | [TIMING] Total Rerun publishing: 4 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 47 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | Frame 246 - Total Time: 21 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 17 ms (80.952%)
mesh_service  |      - Normal estimation: ~11.050 ms (estimate)
mesh_service  |      - TSDF + MC: ~5.100 ms (estimate)
mesh_service  |      - Other: ~0.850 ms (estimate)
mesh_service  |   2. Rerun Publishing: 4 ms (19.048%)
mesh_service  |   3. Cleanup: 0 ms (0.000%)
mesh_service  |   4. Other (metrics, etc): 0 ms
mesh_service  | Performance: 47.619 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 21 ms
mesh_service  | [TIMING] Total message handling: 22 ms
mesh_service  | Received message, size: 1857932 bytes
mesh_service  | [TIMING] msgpack unpacking: 1176 µs
mesh_service  | Message type: 8
mesh_service  | Iterating through msgpack map:
mesh_service  |   Key: timestamp, Value type: 2
mesh_service  |     Timestamp type: 2, value: 40679440141000000
mesh_service  |   Key: keyframe_id, Value type: 2
mesh_service  |   Key: frame_id, Value type: 2
mesh_service  |   Key: pts3d_world, Value type: 7
mesh_service  |   Key: conf_world, Value type: 7
mesh_service  | [MESSAGE TYPE FIX] Message has NO 'type' field, defaulting to 'keyframe.new'
mesh_service  | [MESSAGE TYPE FIX] Message size: 1857932 bytes
mesh_service  | [MESSAGE TYPE FIX] Message fields present: timestamp=yes, keyframe_id=no, shm_key=no, point_count=1155494976
mesh_service  | Parsed message - shm_key: , keyframe_id: , type: keyframe.new, point_count: 1155494976
mesh_service  | [TIMING] Message parsing: 6 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe  via RabbitMQ, shm_key: 
mesh_service  | [DEBUG] Opening shared memory segment: 
mesh_service  | [SHM DEBUG] Opening shared memory: 
mesh_service  | [TIMING] Keyframe handler total: 0 ms
mesh_service  | [SHM DEBUG] shm_open failed, errno: 22 (Invalid argument)
mesh_service  | Failed to open shared memory segment: 
mesh_service  | [TIMING] Total message handling: 1 ms
<LOG>