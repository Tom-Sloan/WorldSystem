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
mesh_service  | [CAMERA NORMAL] Point 7: pos=[5.866,0.408,25.507], normal=[0.224,0.016,0.974], dist=26.176
mesh_service  | [CAMERA NORMAL] Point 8: pos=[5.869,0.402,25.536], normal=[0.224,0.015,0.974], dist=26.205
mesh_service  | [CAMERA NORMAL] Point 9: pos=[5.889,0.393,25.496], normal=[0.225,0.015,0.974], dist=26.170
mesh_service  | [NORMAL ESTIMATION] Provider: 1 (Camera-based (fast))
mesh_service  | [NORMAL ESTIMATION] Time: 1 ms
mesh_service  | [NORMAL ESTIMATION] Points: 48758
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.0000 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 48758, Complexity: 0.4876
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 48758 points
mesh_service  | [MC BOUNDS UPDATE] Point cloud bounds changed, re-initializing TSDF
mesh_service  |   Old bounds: [5.1610,0.1400,25.0770] to [10.9610,3.4900,32.5770]
mesh_service  |   New bounds: [5.1620,0.1716,25.0974] to [11.0064,3.5364,32.7083]
mesh_service  | [SIMPLE TSDF INIT] Initializing TSDF volume:
mesh_service  |   Volume min: [5.1620, 0.1716, 25.0974]
mesh_service  |   Volume max: [11.0064, 3.5364, 32.7083]
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Volume dimensions: [117, 67, 152]
mesh_service  |   Total voxels: 1191528
mesh_service  | SimpleTSDF initialized:
mesh_service  |   Volume dims: 117x67x152
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Memory usage: 9.0906 MB
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Full camera pose matrix (column-major):
mesh_service  |   [    1.0000,     0.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     1.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     1.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     0.0000,     1.0000]
mesh_service  | [TSDF DEBUG] Camera position extracted: [0.0000, 0.0000, 0.0000]
mesh_service  | [TSDF WARNING] Camera position is at origin [0,0,0] - this is likely invalid!
mesh_service  | [TSDF WARNING] This will cause incorrect TSDF carving. Check SLAM3R pose output.
mesh_service  | [TSDF DEBUG] integrate() called with 48758 points
mesh_service  | [TSDF DEBUG] Normal provider: External normals provided
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.1000 m
mesh_service  | [TIMING] TSDF debug copy: 9 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [5.7425, 0.3937, 25.4772]
mesh_service  |   Point 1: [5.7739, 0.3961, 25.4815]
mesh_service  |   Point 2: [5.7898, 0.4068, 25.4661]
mesh_service  |   Point 3: [5.8162, 0.4123, 25.4835]
mesh_service  |   Point 4: [5.8252, 0.4126, 25.4800]
mesh_service  |   Point 5: [5.8310, 0.4075, 25.5030]
mesh_service  |   Point 6: [5.8512, 0.4074, 25.5045]
mesh_service  |   Point 7: [5.8657, 0.4077, 25.5073]
mesh_service  |   Point 8: [5.8687, 0.4022, 25.5360]
mesh_service  |   Point 9: [5.8885, 0.3928, 25.4962]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.0000, Translation X: -103.2394
mesh_service  |   Scale Y: 20.0000, Translation Y: -3.4323
mesh_service  |   Scale Z: 20.0000, Translation Z: -501.9475
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 117x67x152
mesh_service  |   Origin: [5.1620, 0.1716, 25.0974]
mesh_service  |   Voxel size: 0.0500
mesh_service  |   Truncation distance: 0.1000
mesh_service  | [TSDF KERNEL] Point 0: world=[5.742,0.394,25.477] -> voxel=[11.610,4.441,7.596], normal=[0.220,0.015,0.975]
mesh_service  | [TSDF KERNEL] Point 1: world=[5.774,0.396,25.481] -> voxel=[12.238,4.490,7.682], normal=[0.221,0.015,0.975]
mesh_service  | [TSDF KERNEL] Point 2: world=[5.790,0.407,25.466] -> voxel=[12.556,4.703,7.375], normal=[0.222,0.016,0.975]
mesh_service  | [TSDF KERNEL] Point 3: world=[5.816,0.412,25.484] -> voxel=[13.085,4.813,7.723], normal=[0.222,0.016,0.975]
mesh_service  | [TSDF KERNEL] Point 4: world=[5.825,0.413,25.480] -> voxel=[13.264,4.820,7.652], normal=[0.223,0.016,0.975]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[9,2,5], max=[14,7,10], volume_dims=[117,67,152]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[10,2,5], max=[15,7,10], volume_dims=[117,67,152]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[10,2,5], max=[15,7,10], volume_dims=[117,67,152]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[11,2,5], max=[16,7,10], volume_dims=[117,67,152]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[11,2,5], max=[16,7,10], volume_dims=[117,67,152]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1191528
mesh_service  |   Modified voxels: 1190923
mesh_service  |   Weighted voxels: 70424
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 117x67x152
mesh_service  |   Origin: [5.1620, 0.1716, 25.0974]
mesh_service  |   Voxel size: 0.0500
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 140848 active voxels (safety margin from 70424 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 70424 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 31720, weight=3.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 31721, weight=4.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 31722, weight=7.000, pos=9
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24225, weight=3.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24226, weight=4.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24227, weight=2.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24228, weight=2.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24231, weight=1.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24232, weight=2.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24233, weight=1.000, pos=6
mesh_service  | [MC DEBUG] Found 70424 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 
mesh_service  | [MC DEBUG] Classifying 70424 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 24225): coords=[6,6,3]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 24226): coords=[7,6,3]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 24227): coords=[8,6,3]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 24228): coords=[9,6,3]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 24231): coords=[12,6,3]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [-0.090,-0.088,-0.071,-0.073,-0.051,-0.013,-0.053,-0.068]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [-0.088,-0.094,-0.076,-0.071,-0.013,-0.048,-0.063,-0.053]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [-0.094,-0.096,-0.086,-0.076,-0.048,-0.085,-0.020,-0.063]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [-0.096,1.000,-0.094,-0.086,-0.085,-0.092,-0.024,-0.020]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [-0.099,-0.086,-0.083,-0.088,-0.097,-0.073,-0.073,-0.078]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=255, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=255, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=255, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=253, num_verts=3
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=255, num_verts=0
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 1 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 70424
mesh_service  |   Active voxels with triangles: 53410
mesh_service  |   Total vertices to generate: 382755
mesh_service  |   Last vert scan: 382746, orig: 9
mesh_service  |   Last occupied scan: 53409, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 53410 active voxels with 382755 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 27744000
mesh_service  |   Active voxels with triangles: 53410
mesh_service  |   Total vertices needed: 382755
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 53410
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 53410
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=3, out_idx=0, global_voxel=24228, verts=3
mesh_service  | [EXTRACT] idx=6, out_idx=1, global_voxel=24233, verts=6
mesh_service  | [EXTRACT] idx=11, out_idx=2, global_voxel=31724, verts=6
mesh_service  | [EXTRACT] idx=13, out_idx=3, global_voxel=24116, verts=6
mesh_service  | [EXTRACT] idx=19, out_idx=4, global_voxel=64949, verts=3
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 53410
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 53410
mesh_service  |   Output size: 53410
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=3, scan=0
mesh_service  |   [1] verts=6, scan=3
mesh_service  |   [2] verts=6, scan=9
mesh_service  |   [3] verts=6, scan=15
mesh_service  |   [4] verts=3, scan=21
mesh_service  |   [5] verts=3, scan=24
mesh_service  |   [6] verts=6, scan=27
mesh_service  |   [7] verts=9, scan=33
mesh_service  |   [8] verts=9, scan=42
mesh_service  |   [9] verts=9, scan=51
mesh_service  |   Last element: verts=9, scan=382746, total=382755
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x79e552000000
mesh_service  |   d_compressed_global_voxels ptr: 0x79e5252be000
mesh_service  |   d_compressed_verts_scan ptr: 0x79e525326800
mesh_service  |   d_vertex_buffer: 0x79e526000000
mesh_service  |   d_normal_buffer: 0x79e520000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 117x67x152
mesh_service  |   origin: [5.1620, 0.1716, 25.0974]
mesh_service  |   num_active_voxels: 53410
mesh_service  |   voxel_size: 0.0500
mesh_service  |   iso_value: 0.0000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 835, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=24228, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=24233, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: voxel_idx=31724, vertex_offset=9
mesh_service  | [KERNEL] Thread 0: coords=[9,6,3], dims=[117,67,152]
mesh_service  | [KERNEL] Thread 1: coords=[14,6,3], dims=[117,67,152]
mesh_service  | [KERNEL] Thread 2: coords=[17,3,4], dims=[117,67,152]
mesh_service  | [KERNEL] Thread 0: base_idx=24228, max_idx=1191528
mesh_service  | [KERNEL] Thread 1: base_idx=24233, max_idx=1191528
mesh_service  | [KERNEL] Thread 2: base_idx=31724, max_idx=1191528
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=253
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=249
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=249
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=9
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=3
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=9
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 0 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 3 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 12 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 1 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 3 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 12 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 55 µs
mesh_service  | Mesh generation completed in 16ms: 382755 vertices, 127585 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 16 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~12 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 48758
mesh_service  |     - Points/sec: 3047375.0000
mesh_service  |     - Vertices generated: 382755
mesh_service  |     - Faces generated: 127585
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 382755 vertices, 127585 faces in 16ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 1148265
mesh_service  | [DEBUG] Extracting colors for 48758 points
mesh_service  | [DEBUG] Color extraction complete, size: 146274
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 382755 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [5.6163, 0.4716, 25.2474]
mesh_service  | [DEBUG Rerun] Vertex 1: [5.6620, 0.4716, 25.2932]
mesh_service  | [DEBUG Rerun] Vertex 2: [5.6620, 0.5173, 25.2474]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 1 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 48 µs
mesh_service  | [TIMING] Total Rerun publishing: 1 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 27 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 356 - Total Time: 18 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 16 ms (88.8889%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~12.8000 ms (estimate)
mesh_service  |      - Other: ~3.2000 ms (estimate)
mesh_service  |   2. Rerun Publishing: 1 ms (5.5556%)
mesh_service  |   3. Cleanup: 0 ms (0.0000%)
mesh_service  |   4. Other (metrics, etc): 1 ms
mesh_service  | Performance: 55.5556 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 18 ms
mesh_service  | [TIMING] Total message handling: 18 ms
mesh_service  | Received message, size: 322 bytes
mesh_service  | [TIMING] msgpack unpacking: 12 µs
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
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_357, keyframe_id: 357, type: keyframe.new, point_count: 48737
mesh_service  | [TIMING] Message parsing: 5 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 357 via RabbitMQ, shm_key: /slam3r_keyframe_357
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_357
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_357
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 10 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 7 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 48737, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Full pose matrix from shared memory (row-major):
mesh_service  |   [    1.0000,     0.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     1.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     1.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     0.0000,     1.0000]
mesh_service  | [SHM DEBUG] Camera position (translation) from pose: [0.0000, 0.0000, 0.0000]
mesh_service  | [SHM WARNING] Pose matrix is identity - no camera transform!
mesh_service  | [SHM DEBUG] Header bbox: [4.6098, 0.3972, 24.6122] to [11.2771, 14.6253, 49.4714]
mesh_service  | [SHM DEBUG] Calculated total size: 731159
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x79e65089c000
mesh_service  | [TIMING] mmap full: 2 µs
mesh_service  | [TIMING] Total SharedMemory open: 51 µs (0.0510 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 48737 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [TIMING] Spatial deduplication check: 581 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.0000 m/s, points=48737
mesh_service  | [TIMING] Point/color data access: 0 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 48737 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [5.7522, 0.4193, 25.4988]
mesh_service  |   Point 1: [5.7834, 0.4215, 25.5036]
mesh_service  |   Point 2: [5.7994, 0.4320, 25.4887]
mesh_service  |   Point 3: [5.8260, 0.4375, 25.5066]
mesh_service  |   Point 4: [5.8352, 0.4380, 25.5035]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [4.6098, 0.3972, 24.6122]
mesh_service  |   Max: [11.2771, 14.6253, 49.4714]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [4.6098, 0.3972, 24.6122]
mesh_service  |   Max: [11.2771, 14.6253, 49.4714]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 134 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.5578 MB)
mesh_service  | [TIMING] Octree update: 1 ms
mesh_service  | [CAMERA NORMAL PROVIDER] Computing normals for 48737 points with grid 191 x 1 x 1 and block 256
mesh_service  | [CAMERA NORMAL PROVIDER] Kernel launch failed: invalid configuration argument
mesh_service  | [NORMAL ESTIMATION] Failed, falling back to camera-based in TSDF
mesh_service  | [CAMERA NORMAL] Point 0: pos=[5.752,0.419,25.499], normal=[0.220,0.016,0.975], dist=26.143
mesh_service  | [CAMERA NORMAL] Point 1: pos=[5.783,0.421,25.504], normal=[0.221,0.016,0.975], dist=26.155
mesh_service  | [CAMERA NORMAL] Point 2: pos=[5.799,0.432,25.489], normal=[0.222,0.017,0.975], dist=26.144
mesh_service  | [CAMERA NORMAL] Point 3: pos=[5.826,0.437,25.507], normal=[0.223,0.017,0.975], dist=26.167
mesh_service  | [CAMERA NORMAL] Point 4: pos=[5.835,0.438,25.504], normal=[0.223,0.017,0.975], dist=26.166
mesh_service  | [CAMERA NORMAL] Point 5: pos=[5.841,0.433,25.527], normal=[0.223,0.017,0.975], dist=26.190
mesh_service  | [CAMERA NORMAL] Point 6: pos=[5.861,0.433,25.529], normal=[0.224,0.017,0.975], dist=26.197
mesh_service  | [CAMERA NORMAL] Point 7: pos=[5.876,0.433,25.533], normal=[0.224,0.017,0.974], dist=26.204
mesh_service  | [CAMERA NORMAL] Point 8: pos=[5.879,0.428,25.562], normal=[0.224,0.016,0.974], dist=26.233
mesh_service  | [CAMERA NORMAL] Point 9: pos=[5.898,0.418,25.522], normal=[0.225,0.016,0.974], dist=26.198
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.0000 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 48737, Complexity: 0.4874
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 48737 points
mesh_service  | [MC BOUNDS UPDATE] Point cloud bounds changed, re-initializing TSDF
mesh_service  |   Old bounds: [5.1620,0.1716,25.0974] to [11.0120,3.5216,32.6974]
mesh_service  |   New bounds: [5.1681,0.1972,25.1212] to [11.0241,3.5786,32.7700]
mesh_service  | [SIMPLE TSDF INIT] Initializing TSDF volume:
mesh_service  |   Volume min: [5.1681, 0.1972, 25.1212]
mesh_service  |   Volume max: [11.0241, 3.5786, 32.7700]
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Volume dimensions: [117, 68, 153]
mesh_service  |   Total voxels: 1217268
mesh_service  | SimpleTSDF initialized:
mesh_service  |   Volume dims: 117x68x153
mesh_service  |   Voxel size: 0.0500m
mesh_service  |   Memory usage: 9.2870 MB
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Full camera pose matrix (column-major):
mesh_service  |   [    1.0000,     0.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     1.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     1.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     0.0000,     1.0000]
mesh_service  | [TSDF DEBUG] Camera position extracted: [0.0000, 0.0000, 0.0000]
mesh_service  | [TSDF WARNING] Camera position is at origin [0,0,0] - this is likely invalid!
mesh_service  | [TSDF WARNING] This will cause incorrect TSDF carving. Check SLAM3R pose output.
mesh_service  | [TSDF DEBUG] integrate() called with 48737 points
mesh_service  | [TSDF DEBUG] Normal provider: Camera-based fallback (improved carving)
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.1000 m
mesh_service  | [TIMING] TSDF debug copy: 7 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [5.7522, 0.4193, 25.4988]
mesh_service  |   Point 1: [5.7834, 0.4215, 25.5036]
mesh_service  |   Point 2: [5.7994, 0.4320, 25.4887]
mesh_service  |   Point 3: [5.8260, 0.4375, 25.5066]
mesh_service  |   Point 4: [5.8352, 0.4380, 25.5035]
mesh_service  |   Point 5: [5.8409, 0.4332, 25.5266]
mesh_service  |   Point 6: [5.8610, 0.4329, 25.5289]
mesh_service  |   Point 7: [5.8757, 0.4334, 25.5328]
mesh_service  |   Point 8: [5.8788, 0.4279, 25.5618]
mesh_service  |   Point 9: [5.8982, 0.4184, 25.5222]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.0000, Translation X: -103.3630
mesh_service  |   Scale Y: 20.0000, Translation Y: -3.9449
mesh_service  |   Scale Z: 20.0000, Translation Z: -502.4231
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 117x68x153
mesh_service  |   Origin: [5.1681, 0.1972, 25.1212]
mesh_service  |   Voxel size: 0.0500
mesh_service  |   Truncation distance: 0.1000
mesh_service  | [TSDF KERNEL] Point 0: world=[5.752,0.419,25.499] -> voxel=[11.681,4.440,7.552], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[5.783,0.421,25.504] -> voxel=[12.306,4.485,7.650], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[5.799,0.432,25.489] -> voxel=[12.626,4.696,7.351], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[5.826,0.437,25.507] -> voxel=[13.157,4.805,7.708], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[5.835,0.438,25.504] -> voxel=[13.340,4.815,7.647], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[9,2,5], max=[14,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[10,2,5], max=[15,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[10,2,5], max=[15,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[11,2,5], max=[16,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[11,2,5], max=[16,7,10], volume_dims=[117,68,153]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1217268
mesh_service  |   Modified voxels: 1216676
mesh_service  |   Weighted voxels: 71582
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 117x68x153
mesh_service  |   Origin: [5.1681, 0.1972, 25.1212]
mesh_service  |   Voxel size: 0.0500
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 143164 active voxels (safety margin from 71582 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 71582 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24466, weight=3.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 24467, weight=5.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64948, weight=3.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64949, weight=10.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64950, weight=14.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64951, weight=12.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64952, weight=5.000, pos=6
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64953, weight=8.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64954, weight=16.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64955, weight=15.000, pos=9
mesh_service  | [MC DEBUG] Found 71582 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 
mesh_service  | [MC DEBUG] Classifying 71582 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 24466): coords=[13,5,3]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 24467): coords=[14,5,3]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 64948): coords=[13,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 64949): coords=[14,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 64950): coords=[15,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [-0.094,-0.089,-0.096,-0.088,-0.080,-0.085,-0.083,-0.090]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [-0.089,1.000,1.000,-0.096,-0.085,-0.087,-0.086,-0.083]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [0.399,0.092,0.086,0.091,1.000,1.000,1.000,1.000]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [0.092,0.088,0.088,0.086,1.000,1.000,0.094,1.000]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [0.088,0.084,0.076,0.088,1.000,0.096,0.091,0.094]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=255, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=249, num_verts=6
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=0, num_verts=0
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 0 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 71582
mesh_service  |   Active voxels with triangles: 53274
mesh_service  |   Total vertices to generate: 378390
mesh_service  |   Last vert scan: 378390, orig: 0
mesh_service  |   Last occupied scan: 53274, orig: 0
mesh_service  | [MC DEBUG] Generating triangles for 53274 active voxels with 378390 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 27744000
mesh_service  |   Active voxels with triangles: 53274
mesh_service  |   Total vertices needed: 378390
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 53274
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 53274
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=1, out_idx=0, global_voxel=24467, verts=6
mesh_service  | [EXTRACT] idx=8, out_idx=1, global_voxel=64954, verts=3
mesh_service  | [EXTRACT] idx=9, out_idx=2, global_voxel=64955, verts=9
mesh_service  | [EXTRACT] idx=10, out_idx=3, global_voxel=64956, verts=6
mesh_service  | [EXTRACT] idx=11, out_idx=4, global_voxel=64957, verts=6
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 53274
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 53274
mesh_service  |   Output size: 53274
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=6, scan=0
mesh_service  |   [1] verts=3, scan=6
mesh_service  |   [2] verts=9, scan=9
mesh_service  |   [3] verts=6, scan=18
mesh_service  |   [4] verts=6, scan=24
mesh_service  |   [5] verts=9, scan=30
mesh_service  |   [6] verts=6, scan=39
mesh_service  |   [7] verts=6, scan=45
mesh_service  |   [8] verts=6, scan=51
mesh_service  |   [9] verts=3, scan=57
mesh_service  |   Last element: verts=3, scan=378387, total=378390
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x79e552000000
mesh_service  |   d_compressed_global_voxels ptr: 0x79e5252c0200
mesh_service  |   d_compressed_verts_scan ptr: 0x79e525328600
mesh_service  |   d_vertex_buffer: 0x79e526000000
mesh_service  |   d_normal_buffer: 0x79e520000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 117x68x153
mesh_service  |   origin: [5.1681, 0.1972, 25.1212]
mesh_service  |   num_active_voxels: 53274
mesh_service  |   voxel_size: 0.0500
mesh_service  |   iso_value: 0.0000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 833, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=24467, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=64954, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: voxel_idx=64955, vertex_offset=9
mesh_service  | [KERNEL] Thread 0: coords=[14,5,3], dims=[117,68,153]
mesh_service  | [KERNEL] Thread 1: coords=[19,11,8], dims=[117,68,153]
mesh_service  | [KERNEL] Thread 2: coords=[20,11,8], dims=[117,68,153]
mesh_service  | [KERNEL] Thread 0: base_idx=24467, max_idx=1217268
mesh_service  | [KERNEL] Thread 1: base_idx=64954, max_idx=1217268
mesh_service  | [KERNEL] Thread 2: base_idx=64955, max_idx=1217268
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=249
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=2
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=7
mesh_service  | [KERNEL] Thread 0: Starting triangle generation, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: Starting triangle generation, vertex_offset=6
mesh_service  | [KERNEL] Thread 2: Starting triangle generation, vertex_offset=9
mesh_service  | [KERNEL] Thread 0: Writing vertex at idx=0
mesh_service  | [KERNEL] Thread 1: Writing vertex at idx=6
mesh_service  | [KERNEL] Thread 2: Writing vertex at idx=9
mesh_service  | [MC GENTRI] Kernel execution completed
mesh_service  | [MC DIAG] generateTriangles returned
mesh_service  | [TIMING] Triangle generation: 0 ms
mesh_service  | [MC DEBUG] Triangle generation complete
mesh_service  | [TIMING] Output copy (D2H): 3 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 9 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 0 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 3 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 10 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 57 µs
mesh_service  | Mesh generation completed in 15ms: 378390 vertices, 126130 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 15 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~10 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 48737
mesh_service  |     - Points/sec: 3249133.3333
mesh_service  |     - Vertices generated: 378390
mesh_service  |     - Faces generated: 126130
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 378390 vertices, 126130 faces in 15ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 1135170
mesh_service  | [DEBUG] Extracting colors for 48737 points
mesh_service  | [DEBUG] Color extraction complete, size: 146211
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 378390 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [5.9181, 0.4472, 25.3171]
mesh_service  | [DEBUG Rerun] Vertex 1: [5.9181, 0.4972, 25.3172]
mesh_service  | [DEBUG Rerun] Vertex 2: [5.8725, 0.4972, 25.2712]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 2 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 40 µs
mesh_service  | [TIMING] Total Rerun publishing: 2 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 25 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 357 - Total Time: 18 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 15 ms (83.3333%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~12.0000 ms (estimate)
mesh_service  |      - Other: ~3.0000 ms (estimate)
mesh_service  |   2. Rerun Publishing: 2 ms (11.1111%)
mesh_service  |   3. Cleanup: 0 ms (0.0000%)
mesh_service  |   4. Other (metrics, etc): 1 ms
mesh_service  | Performance: 55.5556 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 18 ms
mesh_service  | [TIMING] Total message handling: 18 ms
mesh_service  | Received message, size: 322 bytes
mesh_service  | [TIMING] msgpack unpacking: 17 µs
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
mesh_service  | Parsed message - shm_key: /slam3r_keyframe_358, keyframe_id: 358, type: keyframe.new, point_count: 48718
mesh_service  | [TIMING] Message parsing: 10 µs
mesh_service  | [MESSAGE TYPE FIX] Processing message with type: keyframe.new
mesh_service  | 
mesh_service  | Received keyframe 358 via RabbitMQ, shm_key: /slam3r_keyframe_358
mesh_service  | [DEBUG] Opening shared memory segment: /slam3r_keyframe_358
mesh_service  | [SHM DEBUG] Opening shared memory: /slam3r_keyframe_358
mesh_service  | [SHM DEBUG] shm_open succeeded, fd: 40
mesh_service  | [TIMING] shm_open: 35 µs
mesh_service  | [SHM DEBUG] Mapping header, size: 104
mesh_service  | [SHM DEBUG] SharedKeyframe struct layout:
mesh_service  |   offset of timestamp_ns: 0
mesh_service  |   offset of point_count: 8
mesh_service  |   offset of color_channels: 12
mesh_service  |   offset of pose_matrix: 16
mesh_service  |   offset of bbox: 80
mesh_service  | [TIMING] mmap header: 13 µs
mesh_service  | [SHM DEBUG] Header mapped, point_count: 48718, color_channels: 3
mesh_service  | [SHM DEBUG] Header timestamp: 1000000000
mesh_service  | [SHM DEBUG] Full pose matrix from shared memory (row-major):
mesh_service  |   [    1.0000,     0.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     1.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     1.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     0.0000,     1.0000]
mesh_service  | [SHM DEBUG] Camera position (translation) from pose: [0.0000, 0.0000, 0.0000]
mesh_service  | [SHM WARNING] Pose matrix is identity - no camera transform!
mesh_service  | [SHM DEBUG] Header bbox: [4.6085, 0.4037, 24.6107] to [11.2772, 14.6482, 49.3048]
mesh_service  | [SHM DEBUG] Calculated total size: 730874
mesh_service  | [SHM DEBUG] Remapping with full size
mesh_service  | [SHM DEBUG] Full mapping successful at 0x79e65089c000
mesh_service  | [TIMING] mmap full: 5 µs
mesh_service  | [TIMING] Total SharedMemory open: 107 µs (0.1070 ms)
mesh_service  | [DEBUG] Successfully opened shared memory
mesh_service  | Processing keyframe with 48718 points
mesh_service  | [DEBUG] Creating MeshUpdate object
mesh_service  | [DEBUG] Calling generateIncrementalMesh
mesh_service  | [TIMING] Spatial deduplication check: 165 µs
mesh_service  | [DEBUG] Processing keyframe: spatial_hash=11093822414574, velocity=0.0000 m/s, points=48718
mesh_service  | [TIMING] Point/color data access: 0 µs
mesh_service  | [MESH GEN DEBUG] Checking point cloud bounds for 48718 points
mesh_service  | [MESH GEN DEBUG] First 5 points raw values:
mesh_service  |   Point 0: [5.7511, 0.4249, 25.5006]
mesh_service  |   Point 1: [5.7825, 0.4271, 25.5055]
mesh_service  |   Point 2: [5.7984, 0.4377, 25.4907]
mesh_service  |   Point 3: [5.8250, 0.4431, 25.5083]
mesh_service  |   Point 4: [5.8340, 0.4437, 25.5054]
mesh_service  | [MESH GEN DEBUG] Actual point cloud bounds:
mesh_service  |   Min: [4.6085, 0.4037, 24.6107]
mesh_service  |   Max: [11.2772, 14.6482, 49.3048]
mesh_service  | [MESH GEN DEBUG] Stored bbox in keyframe:
mesh_service  |   Min: [4.6085, 0.4037, 24.6107]
mesh_service  |   Max: [11.2772, 14.6482, 49.3048]
mesh_service  | [TIMING] Point filtering: 0 ms
mesh_service  | [TIMING] GPU memory allocation: 173 µs
mesh_service  | [TIMING] Host to device copy: 0 ms (0.5575 MB)
mesh_service  | [TIMING] Octree update: 0 ms
mesh_service  | [CAMERA NORMAL PROVIDER] Computing normals for 48718 points with grid 191 x 1 x 1 and block 256
mesh_service  | [CAMERA NORMAL PROVIDER] Kernel launch failed: invalid configuration argument
mesh_service  | [NORMAL ESTIMATION] Failed, falling back to camera-based in TSDF
mesh_service  | [CAMERA NORMAL] Point 0: pos=[5.751,0.425,25.501], normal=[0.220,0.016,0.975], dist=26.144
mesh_service  | [CAMERA NORMAL] Point 1: pos=[5.783,0.427,25.506], normal=[0.221,0.016,0.975], dist=26.156
mesh_service  | [CAMERA NORMAL] Point 2: pos=[5.798,0.438,25.491], normal=[0.222,0.017,0.975], dist=26.146
mesh_service  | [CAMERA NORMAL] Point 3: pos=[5.825,0.443,25.508], normal=[0.223,0.017,0.975], dist=26.169
mesh_service  | [CAMERA NORMAL] Point 4: pos=[5.834,0.444,25.505], normal=[0.223,0.017,0.975], dist=26.168
mesh_service  | [CAMERA NORMAL] Point 5: pos=[5.840,0.439,25.529], normal=[0.223,0.017,0.975], dist=26.192
mesh_service  | [CAMERA NORMAL] Point 6: pos=[5.860,0.438,25.531], normal=[0.224,0.017,0.975], dist=26.199
mesh_service  | [CAMERA NORMAL] Point 7: pos=[5.875,0.439,25.535], normal=[0.224,0.017,0.974], dist=26.206
mesh_service  | [CAMERA NORMAL] Point 8: pos=[5.877,0.434,25.564], normal=[0.224,0.017,0.974], dist=26.235
mesh_service  | [CAMERA NORMAL] Point 9: pos=[5.897,0.424,25.525], normal=[0.225,0.016,0.974], dist=26.201
mesh_service  | [ALGORITHM SELECTOR] Selected method: 0 (0=PoissonRecon, 1=MarchingCubes)
mesh_service  | [ALGORITHM SELECTOR] Camera velocity: 0.0000 m/s
mesh_service  | [ALGORITHM SELECTOR] Points: 48718, Complexity: 0.4872
mesh_service  | [ALGORITHM SELECTOR] Starting reconstruction...
mesh_service  | [MC DEBUG] reconstruct() called with 48718 points
mesh_service  | [MC DEBUG] Integrating points into TSDF...
mesh_service  | [TSDF DEBUG] Full camera pose matrix (column-major):
mesh_service  |   [    1.0000,     0.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     1.0000,     0.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     1.0000,     0.0000]
mesh_service  |   [    0.0000,     0.0000,     0.0000,     1.0000]
mesh_service  | [TSDF DEBUG] Camera position extracted: [0.0000, 0.0000, 0.0000]
mesh_service  | [TSDF WARNING] Camera position is at origin [0,0,0] - this is likely invalid!
mesh_service  | [TSDF WARNING] This will cause incorrect TSDF carving. Check SLAM3R pose output.
mesh_service  | [TSDF DEBUG] integrate() called with 48718 points
mesh_service  | [TSDF DEBUG] Normal provider: Camera-based fallback (improved carving)
mesh_service  | [TSDF DEBUG] Integration method: Camera carving with ray-based empty space detection
mesh_service  | [TSDF DEBUG] Truncation distance: 0.1000 m
mesh_service  | [TIMING] TSDF debug copy: 10 µs
mesh_service  | [TSDF DEBUG] First few points (world space):
mesh_service  |   Point 0: [5.7511, 0.4249, 25.5006]
mesh_service  |   Point 1: [5.7825, 0.4271, 25.5055]
mesh_service  |   Point 2: [5.7984, 0.4377, 25.4907]
mesh_service  |   Point 3: [5.8250, 0.4431, 25.5083]
mesh_service  |   Point 4: [5.8340, 0.4437, 25.5054]
mesh_service  |   Point 5: [5.8396, 0.4389, 25.5286]
mesh_service  |   Point 6: [5.8599, 0.4384, 25.5313]
mesh_service  |   Point 7: [5.8745, 0.4390, 25.5351]
mesh_service  |   Point 8: [5.8773, 0.4338, 25.5642]
mesh_service  |   Point 9: [5.8965, 0.4243, 25.5249]
mesh_service  | [TSDF DEBUG] World-to-volume transform (column-major):
mesh_service  |   Scale X: 20.0000, Translation X: -103.3630
mesh_service  |   Scale Y: 20.0000, Translation Y: -3.9449
mesh_service  |   Scale Z: 20.0000, Translation Z: -502.4231
mesh_service  | [TSDF DEBUG] TSDF volume params:
mesh_service  |   Dims: 117x68x153
mesh_service  |   Origin: [5.1681, 0.1972, 25.1212]
mesh_service  |   Voxel size: 0.0500
mesh_service  |   Truncation distance: 0.1000
mesh_service  | [TSDF KERNEL] Point 0: world=[5.751,0.425,25.501] -> voxel=[11.659,4.554,7.588], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 1: world=[5.783,0.427,25.506] -> voxel=[12.288,4.597,7.687], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 2: world=[5.798,0.438,25.491] -> voxel=[12.605,4.809,7.391], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 3: world=[5.825,0.443,25.508] -> voxel=[13.136,4.918,7.743], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 4: world=[5.834,0.444,25.505] -> voxel=[13.317,4.929,7.685], normal=[0.000,0.000,1.000]
mesh_service  | [TSDF KERNEL] Point 0 voxel bounds: min=[9,2,5], max=[14,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 1 voxel bounds: min=[10,2,5], max=[15,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 2 voxel bounds: min=[10,2,5], max=[15,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 3 voxel bounds: min=[11,2,5], max=[16,7,10], volume_dims=[117,68,153]
mesh_service  | [TSDF KERNEL] Point 4 voxel bounds: min=[11,2,5], max=[16,7,10], volume_dims=[117,68,153]
mesh_service  | [TIMING] TSDF integration kernel: 0 ms
mesh_service  | [TIMING] Total TSDF integration: 0 ms
mesh_service  | [MC DEBUG] TSDF integration complete
mesh_service  | [TIMING] TSDF integration (in MC): 0 ms
mesh_service  | [TIMING] TSDF volume check: 0 ms
mesh_service  | [MC DEBUG] TSDF volume stats:
mesh_service  |   Total voxels: 1217268
mesh_service  |   Modified voxels: 1216702
mesh_service  |   Weighted voxels: 75515
mesh_service  | [MC DEBUG] TSDF volume info:
mesh_service  |   Dims: 117x68x153
mesh_service  |   Origin: [5.1681, 0.1972, 25.1212]
mesh_service  |   Voxel size: 0.0500
mesh_service  | [MC DEBUG FIX] Initialized d_active_count to 0
mesh_service  | [MC DEBUG FIX] Allocated buffer for 151030 active voxels (safety margin from 75515 weighted)
mesh_service  | [MC DEBUG] Finding active voxels from 75515 weighted voxels...
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64947, weight=3.000, pos=0
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64948, weight=10.000, pos=1
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64949, weight=26.000, pos=2
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64950, weight=27.000, pos=3
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64951, weight=28.000, pos=4
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64952, weight=12.000, pos=5
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64953, weight=20.000, pos=6
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64954, weight=28.000, pos=7
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64955, weight=30.000, pos=8
mesh_service  | [FIND ACTIVE DEBUG] Found active voxel at idx 64956, weight=26.000, pos=9
mesh_service  | [MC DEBUG] Found 75515 active voxels to process
mesh_service  | [TIMING] Find active voxels: 0 ms
mesh_service  | [MC DEBUG] First few weight values: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 
mesh_service  | [MC DEBUG] Classifying 75515 active voxels...
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 0 (global idx 64947): coords=[12,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 1 (global idx 64948): coords=[13,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 2 (global idx 64949): coords=[14,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 3 (global idx 64950): coords=[15,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Active voxel 4 (global idx 64951): coords=[16,11,8]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0 TSDF values: [0.099,0.185,0.089,0.096,1.000,1.000,1.000,1.000]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1 TSDF values: [0.185,0.086,0.085,0.089,1.000,1.000,0.097,1.000]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2 TSDF values: [0.086,0.080,0.083,0.085,1.000,0.549,0.089,0.097]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3 TSDF values: [0.080,0.084,0.073,0.083,0.549,0.095,0.089,0.089]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4 TSDF values: [0.084,0.089,0.072,0.073,0.095,0.097,0.277,0.089]
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 0: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 1: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 2: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 3: cube_index=0, num_verts=0
mesh_service  | [MC CLASSIFY ACTIVE] Voxel 4: cube_index=0, num_verts=0
mesh_service  | [MC DEBUG] Active voxel classification complete
mesh_service  | [TIMING] Classify active voxels: 1 ms
mesh_service  | [MC DEBUG] Scan results:
mesh_service  |   Active voxels processed: 75515
mesh_service  |   Active voxels with triangles: 54438
mesh_service  |   Total vertices to generate: 380835
mesh_service  |   Last vert scan: 380832, orig: 3
mesh_service  |   Last occupied scan: 54437, orig: 1
mesh_service  | [MC DEBUG] Generating triangles for 54438 active voxels with 380835 vertices
mesh_service  | [MC DIAG] Buffer allocations:
mesh_service  |   Allocated vertices: 5000000
mesh_service  |   Allocated voxels: 27744000
mesh_service  |   Active voxels with triangles: 54438
mesh_service  |   Total vertices needed: 380835
mesh_service  | [MC DIAG] Pre-allocation check passed
mesh_service  | [MC DIAG] Allocated d_compressed_global_voxels size: 54438
mesh_service  | [MC DIAG] Allocated d_compressed_verts size: 54438
mesh_service  | [MC DIAG] extractCompressedDataKernel launched successfully
mesh_service  | [EXTRACT] idx=7, out_idx=0, global_voxel=64954, verts=3
mesh_service  | [EXTRACT] idx=8, out_idx=1, global_voxel=64955, verts=9
mesh_service  | [EXTRACT] idx=9, out_idx=2, global_voxel=64956, verts=6
mesh_service  | [EXTRACT] idx=10, out_idx=3, global_voxel=64957, verts=6
mesh_service  | [EXTRACT] idx=11, out_idx=4, global_voxel=64958, verts=9
mesh_service  | [MC DIAG] Stream synchronized after extract kernel
mesh_service  | [MC DIAG] Allocated d_compressed_verts_scan size: 54438
mesh_service  | [MC DIAG] Before exclusive_scan:
mesh_service  |   Input size: 54438
mesh_service  |   Output size: 54438
mesh_service  | [MC DIAG] exclusive_scan completed successfully
mesh_service  | [MC DIAG] Post-scan CUDA check passed
mesh_service  | [MC DIAG] First few compressed verts and their scan:
mesh_service  |   [0] verts=3, scan=0
mesh_service  |   [1] verts=9, scan=3
mesh_service  |   [2] verts=6, scan=12
mesh_service  |   [3] verts=6, scan=18
mesh_service  |   [4] verts=9, scan=24
mesh_service  |   [5] verts=6, scan=33
mesh_service  |   [6] verts=9, scan=39
mesh_service  |   [7] verts=6, scan=48
mesh_service  |   [8] verts=9, scan=54
mesh_service  |   [9] verts=6, scan=63
mesh_service  |   Last element: verts=3, scan=380832, total=380835
mesh_service  | [MC DIAG] Verifying pointers before generateTriangles:
mesh_service  |   d_tsdf: 0x79e552000000
mesh_service  |   d_compressed_global_voxels ptr: 0x79e5252c8c00
mesh_service  |   d_compressed_verts_scan ptr: 0x79e525333400
mesh_service  |   d_vertex_buffer: 0x79e526000000
mesh_service  |   d_normal_buffer: 0x79e520000000
mesh_service  | [MC DIAG] Calling generateTriangles...
mesh_service  | [MC GENTRI] generateTriangles called with:
mesh_service  |   dims: 117x68x153
mesh_service  |   origin: [5.1681, 0.1972, 25.1212]
mesh_service  |   num_active_voxels: 54438
mesh_service  |   voxel_size: 0.0500
mesh_service  |   iso_value: 0.0000
mesh_service  | [MC GENTRI] Pre-kernel CUDA check passed
mesh_service  | [MC GENTRI] Allocated d_face_count
mesh_service  | [MC GENTRI] Kernel config - grid: 851, block: 64
mesh_service  | [MC GENTRI] Launching generateTrianglesKernel...
mesh_service  | [MC GENTRI] Kernel launched successfully
mesh_service  | [KERNEL] Thread 0: voxel_idx=64954, vertex_offset=0
mesh_service  | [KERNEL] Thread 1: voxel_idx=64955, vertex_offset=3
mesh_service  | [KERNEL] Thread 2: voxel_idx=64956, vertex_offset=12
mesh_service  | [KERNEL] Thread 0: coords=[19,11,8], dims=[117,68,153]
mesh_service  | [KERNEL] Thread 1: coords=[20,11,8], dims=[117,68,153]
mesh_service  | [KERNEL] Thread 2: coords=[21,11,8], dims=[117,68,153]
mesh_service  | [KERNEL] Thread 0: base_idx=64954, max_idx=1217268
mesh_service  | [KERNEL] Thread 1: base_idx=64955, max_idx=1217268
mesh_service  | [KERNEL] Thread 2: base_idx=64956, max_idx=1217268
mesh_service  | [KERNEL] Thread 0: Successfully loaded field values
mesh_service  | [KERNEL] Thread 1: Successfully loaded field values
mesh_service  | [KERNEL] Thread 2: Successfully loaded field values
mesh_service  | [KERNEL] Thread 0: Calculated cube_index=2
mesh_service  | [KERNEL] Thread 1: Calculated cube_index=7
mesh_service  | [KERNEL] Thread 2: Calculated cube_index=15
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
mesh_service  | [TIMING] Output copy (D2H): 3 ms
mesh_service  | 
mesh_service  | [TIMING] Marching Cubes Total: 10 ms
mesh_service  | [TIMING] MC Breakdown:
mesh_service  |   - TSDF integration: 0 ms
mesh_service  |   - TSDF check: 0 ms
mesh_service  |   - Find active: 0 ms
mesh_service  |   - Classify: 1 ms
mesh_service  |   - Triangle gen: 0 ms
mesh_service  |   - Output copy: 3 ms
mesh_service  | [ALGORITHM SELECTOR] Reconstruction succeeded
mesh_service  | [TIMING] Algorithm processing (TSDF + MC): 11 ms
mesh_service  | [TIMING] Color assignment: 0 ms
mesh_service  | [TIMING] GPU cleanup: 71 µs
mesh_service  | Mesh generation completed in 14ms: 380835 vertices, 126945 faces
mesh_service  | 
mesh_service  | [TIMING SUMMARY]
mesh_service  |   Total mesh generation: 14 ms
mesh_service  |   Major components:
mesh_service  |     - Point filtering: ~0 ms
mesh_service  |     - Normal provider: Open3D (quality)
mesh_service  |     - TSDF + Marching Cubes: ~11 ms
mesh_service  |   Performance metrics:
mesh_service  |     - Points processed: 48718
mesh_service  |     - Points/sec: 3479857.1429
mesh_service  |     - Vertices generated: 380835
mesh_service  |     - Faces generated: 126945
mesh_service  | [DEBUG] Mesh generation complete
mesh_service  | Generated mesh with 380835 vertices, 126945 faces in 14ms
mesh_service  | [DEBUG] Recording metrics
mesh_service  | [DEBUG] Checking Rerun: enabled=1, connected=1
mesh_service  | [DEBUG] Keyframe has colors: 1, vertices count: 1142505
mesh_service  | [DEBUG] Extracting colors for 48718 points
mesh_service  | [DEBUG] Color extraction complete, size: 146154
mesh_service  | [TIMING] Color extraction: 0 ms
mesh_service  | [DEBUG] Publishing colored mesh to Rerun
mesh_service  | [DEBUG Rerun] Publishing mesh with 380835 vertices
mesh_service  | [DEBUG Rerun] Vertex 0: [6.1606, 0.7472, 25.5212]
mesh_service  | [DEBUG Rerun] Vertex 1: [6.1681, 0.7639, 25.5212]
mesh_service  | [DEBUG Rerun] Vertex 2: [6.1681, 0.7472, 25.5267]
mesh_service  | [DEBUG] Colored mesh published
mesh_service  | [TIMING] Rerun publish (colored): 2 ms
mesh_service  | [DEBUG] Logging camera pose
mesh_service  | [DEBUG] Camera pose logged
mesh_service  | [TIMING] Camera pose log: 65 µs
mesh_service  | [TIMING] Total Rerun publishing: 2 ms
mesh_service  | [DEBUG] Closing shared memory
mesh_service  | [TIMING] SharedMemory close: 49 µs
mesh_service  | [DEBUG] Shared memory closed
mesh_service  | [TIMING] Cleanup operations: 0 ms
mesh_service  | 
mesh_service  | ========== FRAME PROCESSING TIMING SUMMARY ==========
mesh_service  | [NORMAL PROVIDER STATUS] Open3D (quality)
mesh_service  | Frame 358 - Total Time: 17 ms
mesh_service  | Breakdown:
mesh_service  |   1. Mesh Generation: 14 ms (82.3529%)
mesh_service  |      - Normal provider: Open3D (quality)
mesh_service  |      - TSDF + MC: ~11.2000 ms (estimate)
mesh_service  |      - Other: ~2.8000 ms (estimate)
mesh_service  |   2. Rerun Publishing: 2 ms (11.7647%)
mesh_service  |   3. Cleanup: 0 ms (0.0000%)
mesh_service  |   4. Other (metrics, etc): 1 ms
mesh_service  | Performance: 58.8235 FPS potential
mesh_service  | ===================================================
mesh_service  | 
mesh_service  | [TIMING] Keyframe handler total: 18 ms
mesh_service  | [TIMING] Total message handling: 18 ms

<LOG>

<LOG SLAM3R>
slam3r  | 2025-07-17 12:32:41,368 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,368 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.580]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,390 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,390 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.120, max: 171.212
slam3r  | 2025-07-17 12:32:41,390 - streaming_slam3r - INFO - Returning keyframe result for frame 1765
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - Publishing keyframe 354 with frame_id 1765
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:41,390 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:41,391 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:41,393 - __main__ - INFO - Publishing keyframe 354: 48756 valid points (from 50176 total, 48756 passed confidence)
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:41,393 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:41,394 - shared_memory - INFO - Wrote keyframe 354 to shared memory: /slam3r_keyframe_354
slam3r  | 2025-07-17 12:32:41,394 - shared_memory - INFO - Keyframe 354 bbox: min=(4.63, 0.32, 24.62), max=(11.23, 14.27, 49.01)
slam3r  | 2025-07-17 12:32:41,397 - streaming_slam3r - INFO - Processing frame 1766 with timestamp 119919826063
encoding images: 100%|██████████| 1/1 [00:00<00:00, 103.39it/s]
slam3r  | 2025-07-17 12:32:41,407 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 12:32:41,426 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,426 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.963]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,452 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,452 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.104, max: 172.116
slam3r  | 2025-07-17 12:32:41,474 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,475 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.106]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,500 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,500 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.116, max: 172.300
slam3r  | 2025-07-17 12:32:41,519 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,519 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.540]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,543 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,544 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.116, max: 174.020
slam3r  | 2025-07-17 12:32:41,562 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,562 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 43.106]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,586 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,587 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.102, max: 173.284
slam3r  | 2025-07-17 12:32:41,587 - shared_memory - INFO - Published keyframe 354 to mesh service (48756 points)
slam3r  | 2025-07-17 12:32:41,587 - __main__ - INFO - Successfully published keyframe 354 with 48756 points
slam3r  | 2025-07-17 12:32:41,588 - streaming_slam3r - INFO - Processing frame 1767 with timestamp 119987770077
encoding images: 100%|██████████| 1/1 [00:00<00:00, 99.13it/s]
slam3r  | 2025-07-17 12:32:41,600 - streaming_slam3r - INFO - Processing frame 1768 with timestamp 120055714091
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.47it/s]
slam3r  | 2025-07-17 12:32:41,612 - streaming_slam3r - INFO - Processing frame 1769 with timestamp 120123658105
encoding images: 100%|██████████| 1/1 [00:00<00:00, 110.39it/s]
slam3r  | 2025-07-17 12:32:41,621 - __main__ - INFO - FPS: 14.74 | Frames: 1770 | Keyframes: 354 | Segment frames: 1770
slam3r  | 2025-07-17 12:32:41,668 - streaming_slam3r - INFO - Processing frame 1770 with timestamp 120191602119
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.96it/s]
slam3r  | 2025-07-17 12:32:41,678 - streaming_slam3r - INFO - Frame 1770 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:41,697 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,697 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.479]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,719 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,719 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.087, max: 173.506
slam3r  | 2025-07-17 12:32:41,719 - streaming_slam3r - INFO - Returning keyframe result for frame 1770
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - Publishing keyframe 355 with frame_id 1770
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:41,719 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:41,722 - __main__ - INFO - Publishing keyframe 355: 48751 valid points (from 50176 total, 48751 passed confidence)
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:41,722 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:41,723 - shared_memory - INFO - Wrote keyframe 355 to shared memory: /slam3r_keyframe_355
slam3r  | 2025-07-17 12:32:41,723 - shared_memory - INFO - Keyframe 355 bbox: min=(4.62, 0.34, 24.62), max=(11.24, 14.40, 49.00)
slam3r  | 2025-07-17 12:32:41,723 - shared_memory - INFO - Published keyframe 355 to mesh service (48751 points)
slam3r  | 2025-07-17 12:32:41,723 - __main__ - INFO - Successfully published keyframe 355 with 48751 points
slam3r  | 2025-07-17 12:32:41,736 - streaming_slam3r - INFO - Processing frame 1771 with timestamp 120259546133
encoding images: 100%|██████████| 1/1 [00:00<00:00, 106.94it/s]
slam3r  | 2025-07-17 12:32:41,746 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 12:32:41,765 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,765 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 43.012]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,792 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,792 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.088, max: 173.447
slam3r  | 2025-07-17 12:32:41,812 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,812 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.846]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,841 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,841 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.087, max: 174.311
slam3r  | 2025-07-17 12:32:41,863 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,863 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.462]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,887 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,888 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.082, max: 175.394
slam3r  | 2025-07-17 12:32:41,908 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,908 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.419]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:41,934 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:41,934 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.084, max: 175.233
slam3r  | 2025-07-17 12:32:41,936 - streaming_slam3r - INFO - Processing frame 1772 with timestamp 120327490148
encoding images: 100%|██████████| 1/1 [00:00<00:00, 99.20it/s]
slam3r  | 2025-07-17 12:32:41,947 - streaming_slam3r - INFO - Processing frame 1773 with timestamp 120395434162
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.18it/s]
slam3r  | 2025-07-17 12:32:41,959 - streaming_slam3r - INFO - Processing frame 1774 with timestamp 120463378176
encoding images: 100%|██████████| 1/1 [00:00<00:00, 106.77it/s]
slam3r  | 2025-07-17 12:32:42,007 - streaming_slam3r - INFO - Processing frame 1775 with timestamp 120531322190
encoding images: 100%|██████████| 1/1 [00:00<00:00, 95.84it/s]
slam3r  | 2025-07-17 12:32:42,019 - streaming_slam3r - INFO - Frame 1775 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:42,039 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,039 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.722]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,061 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,061 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.110, max: 175.136
slam3r  | 2025-07-17 12:32:42,061 - streaming_slam3r - INFO - Returning keyframe result for frame 1775
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - Publishing keyframe 356 with frame_id 1775
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:42,061 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:42,064 - __main__ - INFO - Publishing keyframe 356: 48758 valid points (from 50176 total, 48758 passed confidence)
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:42,064 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:42,065 - shared_memory - INFO - Wrote keyframe 356 to shared memory: /slam3r_keyframe_356
slam3r  | 2025-07-17 12:32:42,065 - shared_memory - INFO - Keyframe 356 bbox: min=(4.61, 0.37, 24.61), max=(11.26, 14.50, 49.12)
slam3r  | 2025-07-17 12:32:42,068 - streaming_slam3r - INFO - Processing frame 1776 with timestamp 120000000000
encoding images: 100%|██████████| 1/1 [00:00<00:00, 105.88it/s]
slam3r  | 2025-07-17 12:32:42,078 - streaming_slam3r - INFO - Processing batch of 4 frames
slam3r  | 2025-07-17 12:32:42,098 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,098 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.413]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,123 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,123 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.080, max: 173.971
slam3r  | 2025-07-17 12:32:42,146 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,146 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.359]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,171 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,171 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.082, max: 173.523
slam3r  | 2025-07-17 12:32:42,193 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,193 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.161]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,218 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,218 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.093, max: 174.334
slam3r  | 2025-07-17 12:32:42,237 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,237 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 42.175]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,261 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,261 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.118, max: 172.642
slam3r  | 2025-07-17 12:32:42,262 - streaming_slam3r - INFO - Processing frame 1777 with timestamp 120067944014
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.22it/s]
slam3r  | 2025-07-17 12:32:42,273 - streaming_slam3r - INFO - Processing frame 1778 with timestamp 120135888028
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.49it/s]
slam3r  | 2025-07-17 12:32:42,284 - streaming_slam3r - INFO - Processing frame 1779 with timestamp 120203832042
encoding images: 100%|██████████| 1/1 [00:00<00:00, 101.04it/s]
slam3r  | 2025-07-17 12:32:42,296 - streaming_slam3r - INFO - Processing frame 1780 with timestamp 120271776056
encoding images: 100%|██████████| 1/1 [00:00<00:00, 104.37it/s]
slam3r  | 2025-07-17 12:32:42,307 - streaming_slam3r - INFO - Frame 1780 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:42,325 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,325 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.690]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,347 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,347 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.158, max: 174.733
slam3r  | 2025-07-17 12:32:42,347 - streaming_slam3r - INFO - Returning keyframe result for frame 1780
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - Publishing keyframe 357 with frame_id 1780
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:42,347 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:42,348 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:42,350 - __main__ - INFO - Publishing keyframe 357: 48737 valid points (from 50176 total, 48737 passed confidence)
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:42,350 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:42,351 - shared_memory - INFO - Wrote keyframe 357 to shared memory: /slam3r_keyframe_357
slam3r  | 2025-07-17 12:32:42,351 - shared_memory - INFO - Keyframe 357 bbox: min=(4.61, 0.40, 24.61), max=(11.28, 14.63, 49.47)
slam3r  | 2025-07-17 12:32:42,352 - streaming_slam3r - INFO - Processing frame 1781 with timestamp 120339720070
encoding images: 100%|██████████| 1/1 [00:00<00:00, 91.07it/s]
slam3r  | 2025-07-17 12:32:42,365 - streaming_slam3r - INFO - Processing frame 1782 with timestamp 120407664084
encoding images: 100%|██████████| 1/1 [00:00<00:00, 109.40it/s]
slam3r  | 2025-07-17 12:32:42,375 - streaming_slam3r - INFO - Processing batch of 5 frames
slam3r  | 2025-07-17 12:32:42,394 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,394 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.484]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,421 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,421 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.122, max: 173.946
slam3r  | 2025-07-17 12:32:42,446 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,446 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.229]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,471 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,471 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.129, max: 173.554
slam3r  | 2025-07-17 12:32:42,490 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,491 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.150]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,515 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,515 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.143, max: 174.148
slam3r  | 2025-07-17 12:32:42,534 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,534 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.484]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,558 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,558 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.145, max: 174.603
slam3r  | 2025-07-17 12:32:42,577 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,577 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.671]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,602 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,602 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.143, max: 173.937
slam3r  | 2025-07-17 12:32:42,603 - streaming_slam3r - INFO - Processing frame 1783 with timestamp 120475608098
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.88it/s]
slam3r  | 2025-07-17 12:32:42,614 - streaming_slam3r - INFO - Processing frame 1784 with timestamp 120543552113
encoding images: 100%|██████████| 1/1 [00:00<00:00, 112.64it/s]
slam3r  | 2025-07-17 12:32:42,624 - shared_memory - INFO - Published keyframe 356 to mesh service (48758 points)
slam3r  | 2025-07-17 12:32:42,624 - __main__ - INFO - Successfully published keyframe 357 with 48758 points
slam3r  | 2025-07-17 12:32:42,625 - streaming_slam3r - INFO - Processing frame 1785 with timestamp 120611496127
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.39it/s]
slam3r  | 2025-07-17 12:32:42,635 - streaming_slam3r - INFO - Frame 1785 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:42,654 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,654 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.297]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,676 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,676 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.147, max: 172.175
slam3r  | 2025-07-17 12:32:42,676 - streaming_slam3r - INFO - Returning keyframe result for frame 1785
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - Publishing keyframe 358 with frame_id 1785
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:42,676 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:42,677 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:42,679 - __main__ - INFO - Publishing keyframe 358: 48718 valid points (from 50176 total, 48718 passed confidence)
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:42,679 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:42,680 - shared_memory - INFO - Wrote keyframe 358 to shared memory: /slam3r_keyframe_358
slam3r  | 2025-07-17 12:32:42,680 - shared_memory - INFO - Keyframe 358 bbox: min=(4.61, 0.40, 24.61), max=(11.28, 14.65, 49.30)
slam3r  | 2025-07-17 12:32:42,681 - streaming_slam3r - INFO - Processing frame 1786 with timestamp 120679440141
encoding images: 100%|██████████| 1/1 [00:00<00:00, 95.36it/s]
slam3r  | 2025-07-17 12:32:42,693 - streaming_slam3r - INFO - Processing frame 1787 with timestamp 120747384155
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.53it/s]
slam3r  | 2025-07-17 12:32:42,704 - shared_memory - INFO - Published keyframe 357 to mesh service (48737 points)
slam3r  | 2025-07-17 12:32:42,704 - __main__ - INFO - Successfully published keyframe 358 with 48737 points
slam3r  | 2025-07-17 12:32:42,705 - streaming_slam3r - INFO - Processing frame 1788 with timestamp 120815328169
encoding images: 100%|██████████| 1/1 [00:00<00:00, 111.01it/s]
slam3r  | 2025-07-17 12:32:42,715 - streaming_slam3r - INFO - Processing batch of 5 frames
slam3r  | 2025-07-17 12:32:42,734 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,734 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.706]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,759 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,759 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.125, max: 172.406
slam3r  | 2025-07-17 12:32:42,782 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,782 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.184]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,807 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,807 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.141, max: 174.498
slam3r  | 2025-07-17 12:32:42,826 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,826 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.493]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,851 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,851 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.151, max: 175.054
slam3r  | 2025-07-17 12:32:42,870 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,870 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.111]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,894 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,894 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.149, max: 174.798
slam3r  | 2025-07-17 12:32:42,914 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,915 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.292]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:42,939 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,939 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.231, max: 172.016
slam3r  | 2025-07-17 12:32:42,941 - streaming_slam3r - INFO - Processing frame 1789 with timestamp 120883272183
encoding images: 100%|██████████| 1/1 [00:00<00:00, 110.36it/s]
slam3r  | 2025-07-17 12:32:42,952 - streaming_slam3r - INFO - Processing frame 1790 with timestamp 120951216197
encoding images: 100%|██████████| 1/1 [00:00<00:00, 110.08it/s]
slam3r  | 2025-07-17 12:32:42,962 - streaming_slam3r - INFO - Frame 1790 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:42,981 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:42,981 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.269]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,003 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,003 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.286, max: 172.592
slam3r  | 2025-07-17 12:32:43,003 - streaming_slam3r - INFO - Returning keyframe result for frame 1790
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - Publishing keyframe 359 with frame_id 1790
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:43,003 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:43,004 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:43,006 - __main__ - INFO - Publishing keyframe 359: 48647 valid points (from 50176 total, 48647 passed confidence)
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:43,006 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:43,007 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:43,007 - shared_memory - INFO - Wrote keyframe 359 to shared memory: /slam3r_keyframe_359
slam3r  | 2025-07-17 12:32:43,008 - shared_memory - INFO - Keyframe 359 bbox: min=(4.61, 0.41, 24.62), max=(11.27, 14.78, 49.46)
slam3r  | 2025-07-17 12:32:43,009 - streaming_slam3r - INFO - Processing frame 1791 with timestamp 121019160211
encoding images: 100%|██████████| 1/1 [00:00<00:00, 105.14it/s]
slam3r  | 2025-07-17 12:32:43,021 - streaming_slam3r - INFO - Processing frame 1792 with timestamp 121087104226
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.90it/s]
slam3r  | 2025-07-17 12:32:43,032 - streaming_slam3r - INFO - Processing frame 1793 with timestamp 121155048240
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.91it/s]
slam3r  | 2025-07-17 12:32:43,042 - shared_memory - INFO - Published keyframe 358 to mesh service (48718 points)
slam3r  | 2025-07-17 12:32:43,042 - __main__ - INFO - Successfully published keyframe 359 with 48718 points
slam3r  | 2025-07-17 12:32:43,044 - streaming_slam3r - INFO - Processing frame 1794 with timestamp 121222992254
encoding images: 100%|██████████| 1/1 [00:00<00:00, 106.12it/s]
slam3r  | 2025-07-17 12:32:43,054 - streaming_slam3r - INFO - Processing batch of 5 frames
slam3r  | 2025-07-17 12:32:43,074 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,074 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.043]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,099 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,099 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.270, max: 175.631
slam3r  | 2025-07-17 12:32:43,118 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,118 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.142]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,142 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,142 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.288, max: 177.152
slam3r  | 2025-07-17 12:32:43,161 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,161 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.321]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,186 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,186 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.302, max: 176.964
slam3r  | 2025-07-17 12:32:43,204 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,204 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.670]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,229 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,229 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.294, max: 176.588
slam3r  | 2025-07-17 12:32:43,248 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,248 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 40.825]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,273 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,273 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.293, max: 176.299
slam3r  | 2025-07-17 12:32:43,274 - streaming_slam3r - INFO - Processing frame 1795 with timestamp 121290936268
encoding images: 100%|██████████| 1/1 [00:00<00:00, 110.60it/s]
slam3r  | 2025-07-17 12:32:43,284 - streaming_slam3r - INFO - Frame 1795 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:43,303 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,303 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 41.058]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,325 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,325 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.341, max: 174.847
slam3r  | 2025-07-17 12:32:43,325 - streaming_slam3r - INFO - Returning keyframe result for frame 1795
slam3r  | 2025-07-17 12:32:43,325 - __main__ - INFO - Publishing keyframe 360 with frame_id 1795
slam3r  | 2025-07-17 12:32:43,325 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,325 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,325 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:43,325 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:43,325 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:43,326 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:43,326 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:43,331 - __main__ - INFO - [PCD] Saved point cloud to /debug_output/slam3r_keyframe_000360.ply (48632 points)
slam3r  | 2025-07-17 12:32:43,333 - __main__ - INFO - [PCD] Point cloud bounds - Min: [ 4.6018877   0.43238395 24.609537  ], Max: [11.26578  15.000359 49.94971 ]
slam3r  | 2025-07-17 12:32:43,333 - __main__ - INFO - [PCD] Confidence stats - Min: 12.002, Max: 174.847, Mean: 108.648
slam3r  | 2025-07-17 12:32:43,334 - __main__ - INFO - Publishing keyframe 360: 48632 valid points (from 50176 total, 48632 passed confidence)
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:43,334 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:43,335 - shared_memory - INFO - Wrote keyframe 360 to shared memory: /slam3r_keyframe_360
slam3r  | 2025-07-17 12:32:43,335 - shared_memory - INFO - Keyframe 360 bbox: min=(4.60, 0.43, 24.61), max=(11.27, 15.00, 49.95)
slam3r  | 2025-07-17 12:32:43,336 - streaming_slam3r - INFO - Processing frame 1796 with timestamp 121358880282
encoding images: 100%|██████████| 1/1 [00:00<00:00, 105.20it/s]
slam3r  | 2025-07-17 12:32:43,347 - shared_memory - INFO - Published keyframe 359 to mesh service (48647 points)
slam3r  | 2025-07-17 12:32:43,347 - __main__ - INFO - Successfully published keyframe 360 with 48647 points
slam3r  | 2025-07-17 12:32:43,348 - streaming_slam3r - INFO - Processing frame 1797 with timestamp 121426824296
encoding images: 100%|██████████| 1/1 [00:00<00:00, 106.96it/s]
slam3r  | 2025-07-17 12:32:43,359 - streaming_slam3r - INFO - Processing frame 1798 with timestamp 121494768310
encoding images: 100%|██████████| 1/1 [00:00<00:00, 107.04it/s]
slam3r  | 2025-07-17 12:32:43,370 - streaming_slam3r - INFO - Processing frame 1799 with timestamp 121562712325
encoding images: 100%|██████████| 1/1 [00:00<00:00, 96.97it/s]
slam3r  | 2025-07-17 12:32:43,381 - __main__ - INFO - FPS: 17.05 | Frames: 1800 | Keyframes: 360 | Segment frames: 1800
slam3r  | 2025-07-17 12:32:43,383 - streaming_slam3r - INFO - Processing frame 1800 with timestamp 121630656339
encoding images: 100%|██████████| 1/1 [00:00<00:00, 103.67it/s]
slam3r  | 2025-07-17 12:32:43,393 - streaming_slam3r - INFO - Frame 1800 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:43,412 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,412 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 37.998]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,434 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,434 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.149, max: 175.002
slam3r  | 2025-07-17 12:32:43,434 - streaming_slam3r - INFO - Returning keyframe result for frame 1800
slam3r  | 2025-07-17 12:32:43,434 - __main__ - INFO - Publishing keyframe 361 with frame_id 1800
slam3r  | 2025-07-17 12:32:43,434 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,435 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,435 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:43,435 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:43,435 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:43,435 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:43,435 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:43,437 - __main__ - INFO - Publishing keyframe 361: 48627 valid points (from 50176 total, 48627 passed confidence)
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:43,438 - shared_memory - INFO - Wrote keyframe 361 to shared memory: /slam3r_keyframe_361
slam3r  | 2025-07-17 12:32:43,439 - shared_memory - INFO - Keyframe 361 bbox: min=(4.60, 0.46, 24.61), max=(11.28, 15.22, 50.62)
slam3r  | 2025-07-17 12:32:43,440 - streaming_slam3r - INFO - Processing frame 1801 with timestamp 121698600353
encoding images: 100%|██████████| 1/1 [00:00<00:00, 90.72it/s]
slam3r  | 2025-07-17 12:32:43,452 - streaming_slam3r - INFO - Processing batch of 5 frames
slam3r  | 2025-07-17 12:32:43,470 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,471 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 39.797]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,496 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,496 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.200, max: 177.094
slam3r  | 2025-07-17 12:32:43,515 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,515 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 39.837]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,540 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,540 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.226, max: 176.388
slam3r  | 2025-07-17 12:32:43,559 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,559 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 39.575]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,584 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,584 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.190, max: 177.448
slam3r  | 2025-07-17 12:32:43,603 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,603 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 39.823]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,628 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,628 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.186, max: 176.286
slam3r  | 2025-07-17 12:32:43,646 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,646 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 38.870]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,671 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,672 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.161, max: 177.569
slam3r  | 2025-07-17 12:32:43,673 - streaming_slam3r - INFO - Processing frame 1802 with timestamp 121766544367
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.95it/s]
slam3r  | 2025-07-17 12:32:43,683 - shared_memory - INFO - Published keyframe 360 to mesh service (48632 points)
slam3r  | 2025-07-17 12:32:43,683 - __main__ - INFO - Successfully published keyframe 361 with 48632 points
slam3r  | 2025-07-17 12:32:43,684 - streaming_slam3r - INFO - Processing frame 1803 with timestamp 121834488381
encoding images: 100%|██████████| 1/1 [00:00<00:00, 108.47it/s]
slam3r  | 2025-07-17 12:32:43,695 - streaming_slam3r - INFO - Processing frame 1804 with timestamp 121902432395
encoding images: 100%|██████████| 1/1 [00:00<00:00, 103.95it/s]
slam3r  | 2025-07-17 12:32:43,708 - streaming_slam3r - INFO - Processing frame 1805 with timestamp 121970376409
encoding images: 100%|██████████| 1/1 [00:00<00:00, 106.55it/s]
slam3r  | 2025-07-17 12:32:43,718 - streaming_slam3r - INFO - Frame 1805 is marked as keyframe, processing immediately
slam3r  | 2025-07-17 12:32:43,737 - streaming_slam3r - INFO - I2P produced pts3d_cam with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,738 - streaming_slam3r - INFO - I2P produced conf_cam with shape: torch.Size([1, 224, 224]), range: [1.001, 39.381]
slam3r  | DEBUG: Decoder iteration 0
slam3r  |   final_srcs[-1] shape: torch.Size([1, 1, 196, 768])
slam3r  |   src_pes shape: torch.Size([1, 1, 196, 768])
slam3r  | 2025-07-17 12:32:43,759 - streaming_slam3r - INFO - L2W produced pts3d_world with shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-17 12:32:43,760 - streaming_slam3r - INFO - L2W produced conf_world with shape: torch.Size([1, 224, 224]), min: 1.124, max: 178.846
slam3r  | 2025-07-17 12:32:43,760 - streaming_slam3r - INFO - Returning keyframe result for frame 1805
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - Publishing keyframe 362 with frame_id 1805
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - pts3d_world shape: torch.Size([1, 224, 224, 3]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - conf_world shape: torch.Size([1, 224, 224]), dtype: torch.float32
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - [SLAM3R POSE DEBUG] Keyframe data keys: ['pose', 'pts3d_world', 'conf_world', 'frame_id', 'timestamp', 'is_keyframe', 'rgb_image']
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose in keyframe_data: True
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose type: <class 'numpy.ndarray'>
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - [SLAM3R POSE DEBUG] Pose value: [[1. 0. 0. 0.]
slam3r  |  [0. 1. 0. 0.]
slam3r  |  [0. 0. 1. 0.]
slam3r  |  [0. 0. 0. 1.]]
slam3r  | 2025-07-17 12:32:43,760 - __main__ - INFO - Before reshape - pts3d_np shape: (1, 224, 224, 3), conf_np shape: (1, 224, 224)
slam3r  | 2025-07-17 12:32:43,762 - __main__ - INFO - Publishing keyframe 362: 48695 valid points (from 50176 total, 48695 passed confidence)
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG] Writing pose matrix to shared memory:
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose shape: (4, 4)
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG] Pose matrix:
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    1.0000,     0.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     1.0000,     0.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     1.0000,     0.0000]
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG]   [    0.0000,     0.0000,     0.0000,     1.0000]
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - [SHM SLAM3R DEBUG] Camera position (translation): [0.0000, 0.0000, 0.0000]
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - WARNING - [SHM SLAM3R WARNING] Pose matrix is identity - no camera transform!
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - Wrote keyframe 362 to shared memory: /slam3r_keyframe_362
slam3r  | 2025-07-17 12:32:43,763 - shared_memory - INFO - Keyframe 362 bbox: min=(4.60, 0.48, 24.60), max=(11.28, 15.26, 50.68)
slam3r  | 2025-07-17 12:32:43,765 - streaming_slam3r - INFO - Processing frame 1806 with timestamp 122038320423
encoding images: 100%|██████████| 1/1 [00:00<00:00, 100.45it/s]
slam3r  | 2025-07-17 12:32:43,776 - streaming_slam3r - INFO - Processing frame 1807 with timestamp 122106264438
encoding images: 100%|██████████| 1/1 [00:00<00:00, 99.15it/s]
slam3r  | 2025-07-17 12:32:43,787 - streaming_slam3r - INFO - Processing batch of 5 frames


<LOG SLAM3R>