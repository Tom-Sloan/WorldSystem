You are an expert programmer and excel at debugging issues.
Do not leave TODOs or placeholders.
When I run my application, I get the following error in <LOGS>.

<HISTORY>
We have been trying to move the visualaiztion portion of slam3r_processing to the mesh_service. In the process we were trying to do more (e.g. batching frames). However, we forgot that we can't modify the underlying model.
<HISTORY>

<OUTOUT>
A working version of slam3r_processor.py
<OUTPUT>

<FILES>
Files to review:
/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/app.py
/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/recon.py
The slam3r paper is at https://arxiv.org/html/2412.09401v3
For a working commit of slam3r_processing.py look at https://github.com/Tom-Sloan/WorldSystem/commit/b15afedda8b36cb8423df86b0f4b9e72a23d6b9b
/home/sam3/Desktop/Toms_Workspace/WorldSystem/docker-compose.yml
https://github.com/PKU-VCL-3DV/SLAM3R/tree/main

Files to edit:
/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/slam3r_processor.py
<FILES>

<NOTES>
app.py and recon.py are a working demo from the original slam3r repo.
Use conda 3dreconstruction for local developement (at /home/sam3/anaconda3/envs/3dreconstruction/bin/python), if there are an libraries missing from 3drecontstruction, install them in 3dreconstruction.
Don't build docker containers, ask me to build them.
<NOTES>

<LOGS>
encoding images: 100%|██████████| 1/1 [00:00<00:00, 92.75it/s]
slam3r  | 2025-07-12 08:08:38,012  INFO  I2P window: 5 views, indices: [0, 1, 2]...[3, 4]
slam3r  | 2025-07-12 08:08:38,109  INFO  ref_view 0 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  ref_view 1 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  ref_view 2 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  ref_view 3 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  ref_view 4 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  L2W preparation - selected 5 reference views from 5 keyframes
slam3r  | 2025-07-12 08:08:38,109  INFO  L2W inference with 5 reference views + 1 source view
slam3r  | 2025-07-12 08:08:38,109  INFO  View 0 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 0 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 0 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 1 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 1 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 1 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 2 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 2 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 2 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 3 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 3 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 3 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 4 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 4 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 4 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 5 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 5 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,109  INFO  View 5 pts3d_cam shape: torch.Size([5, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,111  ERROR  Frame processing error: too many values to unpack (expected 4)
slam3r  | Traceback (most recent call last):
slam3r  |   File "/app/./SLAM3R_engine/slam3r_processor.py", line 1321, in on_video_frame_message
slam3r  |     pose, pc, vis = await process_image_with_slam3r(img, ts_ns)
slam3r  |   File "/app/./SLAM3R_engine/slam3r_processor.py", line 1161, in process_image_with_slam3r
slam3r  |     record = _perform_incremental_processing(view, record)
slam3r  |   File "/app/./SLAM3R_engine/slam3r_processor.py", line 905, in _perform_incremental_processing
slam3r  |     output = l2w_inference(l2w_input_views, l2w_model,
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
slam3r  |     return func(*args, **kwargs)
slam3r  |   File "/app/SLAM3R_engine/slam3r/utils/recon_utils.py", line 333, in l2w_inference
slam3r  |     output = l2w_model(input_views, ref_ids=ref_ids)
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
slam3r  |     return self._call_impl(*args, **kwargs)
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
slam3r  |     return forward_call(*args, **kwargs)
slam3r  |   File "/app/SLAM3R_engine/slam3r/models.py", line 593, in forward
slam3r  |     dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats,
slam3r  |   File "/app/SLAM3R_engine/slam3r/models.py", line 333, in _decode_multiview
slam3r  |     ref_outputs = ref_blk(ref_inputs, src_inputs,
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
slam3r  |     return self._call_impl(*args, **kwargs)
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
slam3r  |     return forward_call(*args, **kwargs)
slam3r  |   File "/app/SLAM3R_engine/slam3r/blocks/multiview_blocks.py", line 149, in forward
slam3r  |     Vx, B, Nx, C = xs.shape
slam3r  | ValueError: too many values to unpack (expected 4)
encoding images: 100%|██████████| 1/1 [00:00<00:00, 92.50it/s]
slam3r  | 2025-07-12 08:08:38,125  INFO  I2P window: 5 views, indices: [0, 1, 2]...[3, 4]
slam3r  | 2025-07-12 08:08:38,222  INFO  ref_view 0 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  ref_view 1 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  ref_view 2 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  ref_view 3 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  ref_view 4 img_tokens shape after retrieve: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  L2W preparation - selected 5 reference views from 5 keyframes
slam3r  | 2025-07-12 08:08:38,222  INFO  L2W inference with 5 reference views + 1 source view
slam3r  | 2025-07-12 08:08:38,222  INFO  View 0 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 0 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 0 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 1 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 1 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 1 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 2 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 2 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 2 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 3 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 3 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 3 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 4 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 4 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 4 pts3d_world shape: torch.Size([1, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 5 img_tokens shape: torch.Size([1, 1, 196, 1024])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 5 true_shape shape: torch.Size([1, 2])
slam3r  | 2025-07-12 08:08:38,222  INFO  View 5 pts3d_cam shape: torch.Size([5, 224, 224, 3])
slam3r  | 2025-07-12 08:08:38,224  ERROR  Frame processing error: too many values to unpack (expected 4)
slam3r  | Traceback (most recent call last):
slam3r  |   File "/app/./SLAM3R_engine/slam3r_processor.py", line 1321, in on_video_frame_message
slam3r  |     pose, pc, vis = await process_image_with_slam3r(img, ts_ns)
slam3r  |   File "/app/./SLAM3R_engine/slam3r_processor.py", line 1161, in process_image_with_slam3r
slam3r  |     record = _perform_incremental_processing(view, record)
slam3r  |   File "/app/./SLAM3R_engine/slam3r_processor.py", line 905, in _perform_incremental_processing
slam3r  |     output = l2w_inference(l2w_input_views, l2w_model,
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
slam3r  |     return func(*args, **kwargs)
slam3r  |   File "/app/SLAM3R_engine/slam3r/utils/recon_utils.py", line 333, in l2w_inference
slam3r  |     output = l2w_model(input_views, ref_ids=ref_ids)
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
slam3r  |     return self._call_impl(*args, **kwargs)
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
slam3r  |     return forward_call(*args, **kwargs)
slam3r  |   File "/app/SLAM3R_engine/slam3r/models.py", line 593, in forward
slam3r  |     dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats,
slam3r  |   File "/app/SLAM3R_engine/slam3r/models.py", line 333, in _decode_multiview
slam3r  |     ref_outputs = ref_blk(ref_inputs, src_inputs,
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
slam3r  |     return self._call_impl(*args, **kwargs)
slam3r  |   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
slam3r  |     return forward_call(*args, **kwargs)
slam3r  |   File "/app/SLAM3R_engine/slam3r/blocks/multiview_blocks.py", line 149, in forward
slam3r  |     Vx, B, Nx, C = xs.shape
slam3r  | ValueError: too many values to unpack (expected 4)
<LOGS>