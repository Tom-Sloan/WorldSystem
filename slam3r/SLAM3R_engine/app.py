import argparse
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import subprocess

from recon import get_img_tokens, initialize_scene, adapt_keyframe_stride, i2p_inference_batch, l2w_inference, normalize_views, scene_frame_retrieve
from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Local2WorldModel, Image2PointsModel
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import *

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default="./tmp", help="value for tempfile.tempdir")

    return parser


def extract_frames(video_path: str, fps: float) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%03d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_path
    ]
    subprocess.run(command, check=True)
    return temp_dir

def recon_scene(i2p_model: Image2PointsModel, 
                l2w_model: Local2WorldModel, 
                device, save_dir, video_extraction_fps, 
                input_images_or_video_path, 
                keyframe_stride, window_radius, initial_window_size, confidence_threshold_i2p,
                num_scene_reference_frames, buffer_update_interval, buffer_strategy, buffer_size,
                confidence_threshold_l2w, num_points_to_save):
    print(f"device: {device},\n save_dir: {save_dir},\n video_extraction_fps: {video_extraction_fps},\n keyframe_stride: {keyframe_stride},\n window_radius: {window_radius},\n initial_window_size: {initial_window_size},\n confidence_threshold_i2p: {confidence_threshold_i2p},\n num_scene_reference_frames: {num_scene_reference_frames},\n buffer_update_interval: {buffer_update_interval},\n buffer_strategy: {buffer_strategy},\n buffer_size: {buffer_size},\n confidence_threshold_l2w: {confidence_threshold_l2w},\n num_points_to_save: {num_points_to_save}")
    np.random.seed(42)
    
    # load the imgs or video
    if isinstance(input_images_or_video_path, str):
        input_images_or_video_path = extract_frames(input_images_or_video_path, video_extraction_fps)
    
    dataset = Seq_Data(input_images_or_video_path, to_tensor=True)
    image_data_views = dataset[0][:] 
    total_num_views = len(image_data_views)
    print(f"total_num_views: {total_num_views}")
    # Pre-save the RGB images along with their corresponding masks 
    # in preparation for visualization at last.
    preprocessed_rgb_images = []
    for i in range(len(image_data_views)):
        if image_data_views[i]['img'].shape[0] == 1:
            image_data_views[i]['img'] = image_data_views[i]['img'][0]        
        preprocessed_rgb_images.append(transform_img(dict(img=image_data_views[i]['img'][None]))[...,::-1])
    
    #preprocess data for extracting their img tokens with encoder
    for view in image_data_views:
        view['img'] = torch.tensor(view['img'][None])
        view['true_shape'] = torch.tensor(view['true_shape'][None])
        for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
            if key in view:
                del view[key]
        to_device(view, device=device)
    # pre-extract img tokens by encoder, which can be reused 
    # in the following inference by both i2p and l2w models
    image_shapes, image_features, image_poses = get_img_tokens(image_data_views, i2p_model)    # 300+fps
    print('finish pre-extracting img tokens')

    # re-organize input views for the following inference.
    # Keep necessary attributes only.
    processed_input_views = []
    for i in range(total_num_views):
        processed_input_views.append(dict(label=image_data_views[i]['label'],
                              img_tokens=image_features[i], 
                              true_shape=image_data_views[i]['true_shape'], 
                              img_pos=image_poses[i]))
    
    # decide the stride of sampling keyframes, as well as other related parameters
    if keyframe_stride == -1:
        keyframe_stride_value = adapt_keyframe_stride(processed_input_views, i2p_model, 
                                          win_r = 3,
                                          adapt_min=1,
                                          adapt_max=20,
                                          adapt_stride=1)
    else:
        keyframe_stride_value = keyframe_stride
    
    # initialize the scene with the first several frames
    initial_window_size = min(initial_window_size, total_num_views//keyframe_stride_value)
    assert initial_window_size >= 2, "not enough views for initializing the scene reconstruction"
    initial_point_clouds, initial_confidences, initial_reference_id = initialize_scene(processed_input_views[:initial_window_size*keyframe_stride_value:keyframe_stride_value], 
                                                   i2p_model, 
                                                   winsize=initial_window_size,
                                                   return_ref_id=True) # 5*(1,224,224,3)
    
    # start reconstrution of the whole scene
    initial_frame_count = len(initial_point_clouds)
    per_frame_results = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
    for key in per_frame_results:
        per_frame_results[key] = [None for _ in range(total_num_views)]
    
    registered_confidence_means = [_ for _ in range(total_num_views)]
    
    # set up the world coordinates with the initial window
    for i in range(initial_frame_count):
        per_frame_results['l2w_confs'][i*keyframe_stride_value] = initial_confidences[i][0].to(device)  # 224,224
        registered_confidence_means[i*keyframe_stride_value] = per_frame_results['l2w_confs'][i*keyframe_stride_value].mean().cpu()

    # initialize the buffering set with the initial window
    assert buffer_size <= 0 or buffer_size >= initial_frame_count 
    buffer_frame_ids = [i*keyframe_stride_value for i in range(initial_frame_count)]
    
    # set up the world coordinates with frames in the initial window
    for i in range(initial_frame_count):
        processed_input_views[i*keyframe_stride_value]['pts3d_world'] = initial_point_clouds[i]
        
    initial_confidence_masks = [conf > confidence_threshold_i2p for conf in initial_confidences] # 1,224,224
    normalized_points = normalize_views([view['pts3d_world'] for view in processed_input_views[:initial_frame_count*keyframe_stride_value:keyframe_stride_value]],
                                                initial_confidence_masks)
    for i in range(initial_frame_count):
        processed_input_views[i*keyframe_stride_value]['pts3d_world'] = normalized_points[i]
        # filter out points with low confidence
        processed_input_views[i*keyframe_stride_value]['pts3d_world'][~initial_confidence_masks[i]] = 0       
        per_frame_results['l2w_pcds'][i*keyframe_stride_value] = normalized_points[i]  # 224,224,3

    # recover the pointmap of each view in their local coordinates with the I2P model
    # TODO: batchify
    local_confidence_means = []
    adjacent_frame_distance = keyframe_stride_value
    for current_view_id in tqdm(range(total_num_views), desc="I2P resonstruction"):
        # skip the views in the initial window
        if current_view_id in buffer_frame_ids:
            # trick to mark the keyframe in the initial window
            if current_view_id // keyframe_stride_value == initial_reference_id:
                per_frame_results['i2p_pcds'][current_view_id] = per_frame_results['l2w_pcds'][current_view_id].cpu()
            else:
                per_frame_results['i2p_pcds'][current_view_id] = torch.zeros_like(per_frame_results['l2w_pcds'][current_view_id], device="cpu")
            per_frame_results['i2p_confs'][current_view_id] = per_frame_results['l2w_confs'][current_view_id].cpu()
            continue
        # construct the local window 
        selected_frame_ids = [current_view_id]
        for i in range(1, window_radius+1):
            if current_view_id-i*adjacent_frame_distance >= 0:
                selected_frame_ids.append(current_view_id-i*adjacent_frame_distance)
            if current_view_id+i*adjacent_frame_distance < total_num_views:
                selected_frame_ids.append(current_view_id+i*adjacent_frame_distance)
        local_window_views = [processed_input_views[id] for id in selected_frame_ids]
        reference_frame_id = 0 
        # recover points in the local window, and save the keyframe points and confs
        output = i2p_inference_batch([local_window_views], i2p_model, ref_id=reference_frame_id, 
                                    tocpu=False, unsqueeze=False)['preds']
        #save results of the i2p model
        per_frame_results['i2p_pcds'][current_view_id] = output[reference_frame_id]['pts3d'].cpu() # 1,224,224,3
        per_frame_results['i2p_confs'][current_view_id] = output[reference_frame_id]['conf'][0].cpu() # 224,224

        # construct the input for L2W model        
        processed_input_views[current_view_id]['pts3d_cam'] = output[reference_frame_id]['pts3d'] # 1,224,224,3
        valid_mask = output[reference_frame_id]['conf'] > confidence_threshold_i2p # 1,224,224
        processed_input_views[current_view_id]['pts3d_cam'] = normalize_views([processed_input_views[current_view_id]['pts3d_cam']],
                                                    [valid_mask])[0]
        processed_input_views[current_view_id]['pts3d_cam'][~valid_mask] = 0 

    local_confidence_means = [conf.mean() for conf in per_frame_results['i2p_confs']] # 224,224
    print(f'finish recovering pcds of {len(local_confidence_means)} frames in their local coordinates, with a mean confidence of {torch.stack(local_confidence_means).mean():.2f}')

    # Special treatment: register the frames within the range of initial window with L2W model
    # TODO: batchify
    if keyframe_stride_value > 1:
        max_confidence_mean = -1
        for current_view_id in tqdm(range((initial_frame_count-1)*keyframe_stride_value), desc="pre-registering"):  
            if current_view_id % keyframe_stride_value == 0:
                continue
            # construct the input for L2W model
            local_to_world_input_views = [processed_input_views[current_view_id]] + [processed_input_views[id] for id in buffer_frame_ids]
            # (for defination of ref_ids, see the doc of l2w_model)
            output = l2w_inference(local_to_world_input_views, l2w_model, 
                                   ref_ids=list(range(1,len(local_to_world_input_views))), 
                                   device=device,
                                   normalize=False)
            
            # process the output of L2W model
            processed_input_views[current_view_id]['pts3d_world'] = output[0]['pts3d_in_other_view'] # 1,224,224,3
            confidence_map = output[0]['conf'] # 1,224,224
            per_frame_results['l2w_confs'][current_view_id] = confidence_map[0] # 224,224
            registered_confidence_means[current_view_id] = confidence_map.mean().cpu()
            per_frame_results['l2w_pcds'][current_view_id] = processed_input_views[current_view_id]['pts3d_world']
            
            if registered_confidence_means[current_view_id] > max_confidence_mean:
                max_confidence_mean = registered_confidence_means[current_view_id]
        print(f'finish aligning {(initial_frame_count-1)*keyframe_stride_value} head frames, with a max mean confidence of {max_confidence_mean:.2f}')
        
        # A problem is that the registered_confidence_means of the initial window is generated by I2P model,
        # while the registered_confidence_means of the frames within the initial window is generated by L2W model,
        # so there exists a gap. Here we try to align it.
        max_initial_confidence_mean = -1
        for i in range(initial_frame_count):
            if registered_confidence_means[i*keyframe_stride_value] > max_initial_confidence_mean:
                max_initial_confidence_mean = registered_confidence_means[i*keyframe_stride_value]
        confidence_alignment_factor = max_confidence_mean/max_initial_confidence_mean
        # print(f'align register confidence with a factor {confidence_alignment_factor}')
        for i in range(initial_frame_count):
            per_frame_results['l2w_confs'][i*keyframe_stride_value] *= confidence_alignment_factor
            registered_confidence_means[i*keyframe_stride_value] = per_frame_results['l2w_confs'][i*keyframe_stride_value].mean().cpu()

    # register the rest frames with L2W model
    next_registration_frame_id = (initial_frame_count-1)*keyframe_stride_value+1 # the next frame to be registered
    buffer_selection_milestone = (initial_frame_count-1)*keyframe_stride_value+1 # All frames before milestone have undergone the selection process for entry into the buffering set.
    frames_per_registration_batch = max(1,min((keyframe_stride_value+1)//2, 10))   # how many frames to register in each round
    buffer_update_interval = keyframe_stride_value*buffer_update_interval   # update the buffering set every buffer_update_interval frames
    maximum_buffer_size = buffer_size
    buffer_management_strategy = buffer_strategy
    candidate_frame_counter = len(buffer_frame_ids) # used for the reservoir sampling strategy
    
    pbar = tqdm(total=total_num_views, desc="registering")
    pbar.update(next_registration_frame_id-1)

    del i
    while next_registration_frame_id < total_num_views:
        batch_start_id = next_registration_frame_id
        batch_end_id = min(batch_start_id+frames_per_registration_batch, total_num_views)-1  # the last frame to be registered in this round

        # select sccene frames in the buffering set to work as a global reference
        candidate_reference_ids = buffer_frame_ids
        reference_views, selected_pool_ids = scene_frame_retrieve(
            [processed_input_views[i] for i in candidate_reference_ids], 
            processed_input_views[batch_start_id:batch_start_id+frames_per_registration_batch:2], 
            i2p_model, sel_num=num_scene_reference_frames, 
            # cand_recon_confs=[per_frame_results['l2w_confs'][i] for i in candidate_reference_ids],
            depth=2)
        
        # register the source frames in the local coordinates to the world coordinates with L2W model
        local_to_world_input_views = reference_views + processed_input_views[batch_start_id:batch_end_id+1]
        total_input_view_count = len(reference_views) + batch_end_id - batch_start_id + 1
        assert total_input_view_count == len(local_to_world_input_views)
        
        output = l2w_inference(local_to_world_input_views, l2w_model, 
                               ref_ids=list(range(len(reference_views))), 
                               device=device,
                               normalize=False)
    
        # process the output of L2W model
        source_ids_in_batch = [id+len(reference_views) for id in range(batch_end_id-batch_start_id+1)]  # the ids of src views in the local window
        source_ids_global = [id for id in range(batch_start_id, batch_end_id+1)]    #the ids of src views in the whole dataset
        successfully_registered_count = 0
        for id in range(len(source_ids_global)):
            batch_output_index = source_ids_in_batch[id] # the id of the output in the output list
            current_view_id = source_ids_global[id]    # the id of the view in all views
            confidence_map = output[batch_output_index]['conf'] # 1,224,224
            processed_input_views[current_view_id]['pts3d_world'] = output[batch_output_index]['pts3d_in_other_view'] # 1,224,224,3
            per_frame_results['l2w_confs'][current_view_id] = confidence_map[0]
            registered_confidence_means[current_view_id] = confidence_map[0].mean().cpu()
            per_frame_results['l2w_pcds'][current_view_id] = processed_input_views[current_view_id]['pts3d_world']
            successfully_registered_count += 1
        # TODO:refine scene frames together
        # for j in range(1, total_input_view_count):
            # views[i-j]['pts3d_world'] = output[total_input_view_count-1-j]['pts3d'].permute(0,3,1,2)

        next_registration_frame_id += successfully_registered_count
        pbar.update(successfully_registered_count) 
        
        # update the buffering set
        if next_registration_frame_id - buffer_selection_milestone >= buffer_update_interval:  
            while(next_registration_frame_id - buffer_selection_milestone >= keyframe_stride_value):
                candidate_frame_counter += 1
                buffer_is_full = maximum_buffer_size > 0 and len(buffer_frame_ids) >= maximum_buffer_size
                should_insert_frame = (not buffer_is_full) or ((buffer_management_strategy == 'fifo') or 
                                                  (buffer_management_strategy == 'reservoir' and np.random.rand() < maximum_buffer_size/candidate_frame_counter))
                if not should_insert_frame: 
                    buffer_selection_milestone += keyframe_stride_value
                    continue
                # Use offest to ensure the selected view is not too close to the last selected view
                # If the last selected view is 0, 
                # the next selected view should be at least keyframe_stride_value*3//4 frames away
                candidate_start_offset = max(0, buffer_frame_ids[-1]+keyframe_stride_value*3//4 - buffer_selection_milestone)
                    
                # get the mean confidence of the candidate views
                candidate_reconstruction_confidences = torch.stack([registered_confidence_means[i]
                                         for i in range(buffer_selection_milestone+candidate_start_offset, buffer_selection_milestone+keyframe_stride_value)])
                candidate_local_confidences = torch.stack([local_confidence_means[i]
                                         for i in range(buffer_selection_milestone+candidate_start_offset, buffer_selection_milestone+keyframe_stride_value)])
                # normalize the confidence to [0,1], to avoid overconfidence
                candidate_reconstruction_confidences = (candidate_reconstruction_confidences - 1)/candidate_reconstruction_confidences # transform to sigmoid
                candidate_local_confidences = (candidate_local_confidences - 1)/candidate_local_confidences
                # the final confidence is the product of the two kinds of confidences
                combined_candidate_confidences = candidate_reconstruction_confidences*candidate_local_confidences
                
                highest_confidence_index = combined_candidate_confidences.argmax().item()
                highest_confidence_index += candidate_start_offset
                frame_id_to_add_to_buffer = buffer_selection_milestone + highest_confidence_index
                buffer_frame_ids.append(frame_id_to_add_to_buffer)
                # print(f"add ref view {frame_id_to_add_to_buffer}")                
                # since we have inserted a new frame, overflow must happen when buffer_is_full is True
                if buffer_is_full:
                    if buffer_management_strategy == 'reservoir':
                        buffer_frame_ids.pop(np.random.randint(maximum_buffer_size))
                    elif buffer_management_strategy == 'fifo':
                        buffer_frame_ids.pop(0)
                # print(next_registration_frame_id, buffer_frame_ids)
                buffer_selection_milestone += keyframe_stride_value
        # transfer the data to cpu if it is not in the buffering set, to save gpu memory
        for i in range(next_registration_frame_id):
            to_device(processed_input_views[i], device=device if i in buffer_frame_ids else 'cpu')
    
    pbar.close()
    
    low_confidence_frames = {}
    for i, conf in enumerate(registered_confidence_means):
        if conf < 10:
            low_confidence_frames[i] = conf.item()
    print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confidence_means).mean().item():.2f}')
    print(f"{len(low_confidence_frames)} views with low confidence: ", {key:round(low_confidence_frames[key],2) for key in low_confidence_frames.keys()})

    per_frame_results['rgb_imgs'] = preprocessed_rgb_images

    save_path = get_model_from_scene(per_frame_res=per_frame_results, 
                                     save_dir=save_dir, 
                                     num_points_save=num_points_to_save, 
                                     conf_thres_res=confidence_threshold_l2w)

    return save_path, per_frame_results
    
    
def get_model_from_scene(per_frame_res, save_dir, 
                         num_points_save=200000, 
                         conf_thres_res=3, 
                         valid_masks=None
                        ):  
        
    # collect the registered point clouds and rgb colors
    pcds = []
    rgbs = []
    pred_frame_num = len(per_frame_res['l2w_pcds'])
    registered_confs = per_frame_res['l2w_confs']   
    registered_pcds = per_frame_res['l2w_pcds']
    rgb_imgs = per_frame_res['rgb_imgs']
    for i in range(pred_frame_num):
        registered_pcd = to_numpy(registered_pcds[i])
        if registered_pcd.shape[0] == 3:
            registered_pcd = registered_pcd.transpose(1,2,0)
        registered_pcd = registered_pcd.reshape(-1,3)
        rgb = rgb_imgs[i].reshape(-1,3)
        pcds.append(registered_pcd)
        rgbs.append(rgb)
        
    res_pcds = np.concatenate(pcds, axis=0)
    res_rgbs = np.concatenate(rgbs, axis=0)
    
    pts_count = len(res_pcds)
    valid_ids = np.arange(pts_count)
    
    # filter out points with gt valid masks
    if valid_masks is not None:
        valid_masks = np.stack(valid_masks, axis=0).reshape(-1)
        # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
    else:
        valid_masks = np.ones(pts_count, dtype=bool)
    
    # filter out points with low confidence
    if registered_confs is not None:
        conf_masks = []
        for i in range(len(registered_confs)):
            conf = registered_confs[i]
            conf_mask = (conf > conf_thres_res).reshape(-1).cpu() 
            conf_masks.append(conf_mask)
        conf_masks = np.array(torch.cat(conf_masks))
        valid_ids = valid_ids[conf_masks&valid_masks]
        print('ratio of points filered out: {:.2f}%'.format((1.-len(valid_ids)/pts_count)*100))
    
    # sample from the resulting pcd consisting of all frames
    n_samples = min(num_points_save, len(valid_ids))
    print(f"resampling {n_samples} points from {len(valid_ids)} points")
    sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
    sampled_pts = res_pcds[sampled_idx]
    sampled_rgbs = res_rgbs[sampled_idx]
    sampled_pts[:, :2] *= -1 # flip the x,y axis for better visualization
    
    save_name = f"recon.glb"
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=sampled_pts, colors=sampled_rgbs/255.))
    save_path = join(save_dir, save_name)
    scene.export(save_path)

    return save_path

def display_inputs(images):
    img_label = "Click or use the left/right arrow keys to browse images", 

    if images is None or len(images) == 0:
        return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                gradio.update(value=None, visible=False, scale=2, height=300,)]  

    if isinstance(images, str): 
        file_path = images
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        if any(file_path.endswith(ext) for ext in video_extensions):
            return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                gradio.update(value=file_path, autoplay=True, visible=True, scale=2, height=300,)]
        else:
            return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                    gradio.update(value=None, visible=False, scale=2, height=300,)] 
            
    return [gradio.update(label=img_label, value=images, visible=True, selected_index=0, scale=2, preview=True, height=300,),
            gradio.update(value=None, visible=False, scale=2, height=300,)]

def change_inputfile_type(input_type):
    #刷新gradio.File中的文件
    if input_type == "directory":
        inputfiles = gradio.File(file_count="directory", file_types=["image"],
                                 scale=2,
                                 value=[],
                                 label="Select a directory containing images")
        video_extract_fps = gradio.Number(value=5,
                                          scale=0,
                                          interactive=True,
                                          visible=False,
                                          label="fps for extracting frames from video")
    elif input_type == "images":
        inputfiles = gradio.File(file_count="multiple", file_types=["image"],
                                 scale=2,
                                 value=[],
                                 label="Upload multiple images")
        video_extract_fps = gradio.Number(value=5,
                                          scale=0,
                                          interactive=True,
                                          visible=False,
                                          label="fps for extracting frames from video")
    elif input_type == "video":
        inputfiles = gradio.File(file_count="single", file_types=["video"],
                                 scale=2,
                                 value=None,
                                 label="Upload a mp4 video")
        video_extract_fps = gradio.Number(value=5,
                                          interactive=True,
                                          scale=1,
                                          visible=True,
                                          label="fps for extracting frames from video")
    return inputfiles, video_extract_fps
    
def change_kf_stride_type(kf_stride, inputfiles, win_r):
    max_kf_stride = 10
    if kf_stride == "auto":
        kf_stride_fix = gradio.Slider(value=-1,minimum=-1, maximum=-1, step=1, 
                                      visible=False, interactive=True, 
                                      label="stride between keyframes",
                                      info="For I2P reconstruction!")
    elif kf_stride == "manual setting":
        kf_stride_fix = gradio.Slider(value=1,minimum=1, maximum=max_kf_stride, step=1, 
                                      visible=True, interactive=True, 
                                      label="stride between keyframes",
                                      info="For I2P reconstruction!")
    return kf_stride_fix

def change_buffer_strategy(buffer_strategy):
    if buffer_strategy == "reservoir" or buffer_strategy == "fifo":
        buffer_size = gradio.Number(value=100, precision=0, minimum=1,
                                    interactive=True, 
                                    visible=True,
                                    label="size of the buffering set",
                                    info="For L2W reconstruction!")
    elif buffer_strategy == "unbounded":
        buffer_size = gradio.Number(value=10000, precision=0, minimum=1,
                                    interactive=True, 
                                    visible=False,
                                    label="size of the buffering set",
                                    info="For L2W reconstruction!")
    return buffer_size

def main_demo(i2p_model, l2w_model, device, tmpdirname, server_name, server_port):
    """
    Main demo function that creates and launches the SLAM3R Gradio interface.
    
    Args:
        i2p_model: Image2Points model for local 3D reconstruction
        l2w_model: Local2World model for global registration
        device: Device to run models on (CPU/GPU)
        tmpdirname: Temporary directory for storing results
        server_name: Server hostname/IP
        server_port: Server port number
    """
    # Create a partial function with fixed model and device parameters
    recon_scene_func = functools.partial(recon_scene, i2p_model, l2w_model, device)
    
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="SLAM3R Demo") as demo:
        # State variables to persist data between interactions
        # per_frame_res stores reconstruction results to avoid re-running inference
        per_frame_res = gradio.State(None)
        # tmpdir_name stores the temporary directory path
        tmpdir_name = gradio.State(tmpdirname)
        
        # Main header
        gradio.HTML('<h2 style="text-align: center;">SLAM3R Demo</h2>')
        
        with gradio.Column():
            # Input file selection section
            with gradio.Row():
                # Input type selector (directory, images, or video)
                input_type = gradio.Dropdown([ "directory", "images", "video"],
                                             scale=1,
                                             value='directory', label="select type of input files")
                # FPS setting for video frame extraction (hidden by default)
                video_extract_fps = gradio.Number(value=5,
                                                  scale=0,
                                                  interactive=True,
                                                  visible=False,
                                                  label="fps for extracting frames from video")
                # File upload component (defaults to directory selection)
                inputfiles = gradio.File(file_count="directory", file_types=["image"],
                                         scale=2,
                                         height=200,
                                         label="Select a directory containing images")
              
                # Image gallery for previewing uploaded images
                image_gallery = gradio.Gallery(label="Click or use the left/right arrow keys to browse images",
                                            visible=False,
                                            selected_index=0,
                                            preview=True, 
                                            height=300,
                                            scale=2)
                # Video player for uploaded videos
                video_gallery = gradio.Video(label="Uploaded Video",
                                            visible=False,
                                            height=300,
                                            scale=2)

            # I2P (Image2Points) reconstruction parameters
            with gradio.Row():
                # Keyframe stride selection (auto or manual)
                kf_stride = gradio.Dropdown(["auto", "manual setting"], label="how to choose stride between keyframes",
                                           value='auto', interactive=True,  
                                           info="For I2P reconstruction!")
                # Manual keyframe stride slider (hidden when auto is selected)
                kf_stride_fix = gradio.Slider(value=-1, minimum=-1, maximum=-1, step=1, 
                                              visible=False, interactive=True, 
                                              label="stride between keyframes",
                                              info="For I2P reconstruction!")
                # Window radius for local reconstruction
                win_r = gradio.Number(value=5, precision=0, minimum=1, maximum=200,
                                      interactive=True, 
                                      label="the radius of the input window",
                                      info="For I2P reconstruction!")
                # Number of frames for scene initialization
                initial_winsize = gradio.Number(value=5, precision=0, minimum=2, maximum=200,
                                      interactive=True, 
                                      label="the number of frames for initialization",
                                      info="For I2P reconstruction!")
                # Confidence threshold for I2P model
                conf_thres_i2p = gradio.Slider(value=1.5, minimum=1., maximum=10,
                                      interactive=True, 
                                      label="confidence threshold for the i2p model",
                                      info="For I2P reconstruction!")
            
            # L2W (Local2World) reconstruction parameters
            with gradio.Row():
                # Number of scene frames to use as reference
                num_scene_frame = gradio.Slider(value=10, minimum=1., maximum=100, step=1,
                                      interactive=True, 
                                      label="the number of scene frames for reference",
                                      info="For L2W reconstruction!")
                # Buffer management strategy
                buffer_strategy = gradio.Dropdown(["reservoir", "fifo","unbounded"], 
                                           value='reservoir', interactive=True,  
                                           label="strategy for buffer management",
                                           info="For L2W reconstruction!")
                # Buffer size for storing reference frames
                buffer_size = gradio.Number(value=100, precision=0, minimum=1,
                                      interactive=True, 
                                      visible=True,
                                      label="size of the buffering set",
                                      info="For L2W reconstruction!")
                # Interval for updating the buffer
                update_buffer_intv = gradio.Number(value=1, precision=0, minimum=1,
                                      interactive=True, 
                                      label="the interval of updating the buffering set",
                                      info="For L2W reconstruction!")
            
            # Main execution button
            run_btn = gradio.Button("Run")

            # Post-processing parameters
            with gradio.Row():
                # Confidence threshold for final result filtering
                conf_thres_l2w = gradio.Slider(value=12, minimum=1., maximum=100,
                                      interactive=True, 
                                      label="confidence threshold for the result",
                                      )
                # Number of points to sample from final reconstruction
                num_points_save = gradio.Number(value=1000000, precision=0, minimum=1,
                                      interactive=True, 
                                      label="number of points sampled from the result",
                                      )

            # 3D model viewer for displaying reconstruction results
            outmodel = gradio.Model3D(height=500,
                                      clear_color=(0.,0.,0.,0.3)) 
            
            # Event handlers for interactive components
            # Update display when input files change
            inputfiles.change(display_inputs,
                                inputs=[inputfiles],
                                outputs=[image_gallery, video_gallery])
            # Update file input type and show/hide related components
            input_type.change(change_inputfile_type,
                                inputs=[input_type],
                                outputs=[inputfiles, video_extract_fps])
            # Show/hide manual keyframe stride slider based on selection
            kf_stride.change(change_kf_stride_type,
                                inputs=[kf_stride, inputfiles, win_r],
                                outputs=[kf_stride_fix])
            # Update buffer size visibility based on strategy
            buffer_strategy.change(change_buffer_strategy,
                                inputs=[buffer_strategy],
                                outputs=[buffer_size])
            # Main reconstruction execution
            run_btn.click(fn=recon_scene_func,
                          inputs=[tmpdir_name, video_extract_fps,
                                  inputfiles, kf_stride_fix, win_r, initial_winsize, conf_thres_i2p,
                                  num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size,
                                  conf_thres_l2w, num_points_save],
                          outputs=[outmodel, per_frame_res])
            # Update 3D model when confidence threshold changes
            conf_thres_l2w.release(fn=get_model_from_scene,
                                 inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w],
                                 outputs=outmodel)
            # Update 3D model when number of points changes
            num_points_save.change(fn=get_model_from_scene,
                            inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w],
                            outputs=outmodel)

    # Launch the Gradio interface
    demo.launch(share=False, server_name=server_name, server_port=server_port)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'
    
    i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
    l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
    i2p_model.to(args.device)
    l2w_model.to(args.device)
    i2p_model.eval()
    l2w_model.eval()

    # slam3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='slam3r_gradio_demo') as tmpdirname:
        main_demo(i2p_model, l2w_model, args.device, tmpdirname, server_name, args.server_port)
