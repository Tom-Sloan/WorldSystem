# SLAM3R

<p align="center">
  <h2 align="center">[CVPR 2025 Highlight] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos</h2>
 <p align="center">
    <a href="https://ly-kc.github.io/">Yuzheng Liu*</a>
    ·
    <a href="https://siyandong.github.io/">Siyan Dong*</a>
    ·
    <a href="https://ffrivera0.github.io/">Shuzhe Wang</a>
    ·
    <a href="https://yd-yin.github.io/">Yingda Yin</a>
    ·
    <a href="https://yanchaoyang.github.io/">Yanchao Yang</a>
    ·
    <a href="https://fqnchina.github.io/">Qingnan Fan</a>
    ·
    <a href="https://baoquanchen.info/">Baoquan Chen</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2412.09401">Paper</a> | <a href="">Poster</a> </h3>
  <div align="center"></div>
</p>

<div align="center">
  <img src="./media/replica.gif" width="49%" /> 
  <img src="./media/wild.gif" width="49%" />
</div>

<p align="center">
<strong>SLAM3R</strong> is a real-time dense scene reconstruction system that regresses 3D points from video frames using feed-forward neural networks, without explicitly estimating camera parameters. It's part of the WorldSystem project for real-time 3D reconstruction and visualization of drone-based room mapping.
</p>
<be>


## News

* **2025-04:** SLAM3R is reported by [机器之心(Chinese)](https://mp.weixin.qq.com/s/fK5vJwbogcfwoduI9FuQ6w) 

* **2025-04:** 🎉 SLAM3R is selected as a **highlight paper** in CVPR 2025 and **Top1 paper** in China3DV 2025.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Demo](#demo)
- [Gradio interface](#gradio-interface)
- [Docker Usage](#docker-usage)
- [Evaluation on the Replica dataset](#evaluation-on-the-replica-dataset)
- [Training](#training)
- [Real-time Processing](#real-time-processing)
- [Performance Optimization](#performance-optimization)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Project Overview

SLAM3R is part of the WorldSystem project - a real-time RGB-based dense scene reconstruction system that regresses 3D points from video frames using feed-forward neural networks, without explicitly estimating camera parameters. It processes video streams from drones to create 3D reconstructions for AR/VR applications.

## Architecture

### Neural Network Pipeline
1. **Image-to-Points (I2P)**: Processes RGB frames → local 3D point clouds
2. **Local-to-World (L2W)**: Transforms local points → global world coordinates

### Key Components
- `slam3r_processor.py`: RabbitMQ service entry point for real-time processing
- `recon.py`: Offline reconstruction pipeline
- `app.py`: Gradio web interface
- `slam3r/models.py`: Core model definitions (Multiview3D)
- `slam3r/inference.py`: Inference logic and keyframe management

### Real-time Processing Flow
```
RabbitMQ (frames) → Tokenization → I2P Inference → L2W Registration →
Keyframe Selection → Point Accumulation → Mesh Generation → Visualization
```

## Installation

### Local Development

1. Clone SLAM3R
```bash
git clone https://github.com/PKU-VCL-3DV/SLAM3R.git
cd SLAM3R/SLAM3R_engine
```

2. Prepare environment
```bash
conda create -n slam3r python=3.11 cmake=3.14.0
conda activate slam3r 
# install torch according to your cuda version
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# optional: install additional packages to support visualization and data preprocessing
pip install -r requirements_optional.txt  # For Open3D mesh generation
```

3. Optional: Accelerate SLAM3R with XFormers and custom cuda kernels for RoPE
```bash
# install XFormers according to your pytorch version, see https://github.com/facebookresearch/xformers
pip install xformers==0.0.28.post2
# compile cuda kernels for RoPE
# if the compilation fails, try the proposed solution: https://github.com/CUT3R/CUT3R/issues/7.
cd slam3r/pos_embed/curope/
python setup.py build_ext --inplace
cd ../../../
```

4. Optional: Download the SLAM3R checkpoints for the [Image-to-Points](https://huggingface.co/siyan824/slam3r_i2p) and [Local-to-World](https://huggingface.co/siyan824/slam3r_l2w) models through HuggingFace 
```bash
from slam3r.models import Image2PointsModel, Local2WorldModel
Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
```
The pre-trained model weights will automatically download when running the demo and evaluation code below. 

## Docker Usage

### Build and Run
```bash
# Build SLAM3R service
docker-compose build slam3r

# Run SLAM3R service  
docker-compose up slam3r

# Run without SLAM3R (if using other SLAM services)
docker compose up --detach $(docker compose config --services | grep -v slam3r)
```

## Demo

### Replica dataset
To run our demo on Replica dataset, download the sample scene [here](https://drive.google.com/file/d/1NmBtJ2A30qEzdwM0kluXJOp2d1Y4cRcO/view?usp=drive_link) and unzip it to `./data/Replica_demo/`. Then run the following command to reconstruct the scene from the video images 

 ```bash
 bash scripts/demo_replica.sh
 ```

The results will be stored at `./results/` by default.

### Self-captured outdoor data
We also provide a set of images extracted from an in-the-wild captured video. Download it [here](https://drive.google.com/file/d/1FVLFXgepsqZGkIwg4RdeR5ko_xorKyGt/view?usp=drive_link) and unzip it to `./data/wild/`.  

Set the required parameter in this [script](./scripts/demo_wild.sh), and then run SLAM3R by using the following command
 
 ```bash
 bash scripts/demo_wild.sh
 ```

When `--save_preds` is set in the script, the per-frame prediction for reconstruction will be saved at `./results/TEST_NAME/preds/`. Then you can visualize the incremental reconstruction process with the following command

 ```bash
 bash scripts/demo_vis_wild.sh
 ```

A Open3D window will appear after running the script. Please click `space key` to record the adjusted rendering view and close the window. The code will then do the rendering of the incremental reconstruction.

You can run SLAM3R on your self-captured video with the steps above. Here are [some tips](./docs/recon_tips.md) for it


## Gradio interface
We also provide a Gradio interface, where you can upload a directory, a video or specific images to perform the reconstruction. After setting the reconstruction parameters, you can click the 'Run' button to start the process. Modifying the visualization parameters at the bottom allows you to directly display different visualization results without rerunning the inference.

The interface can be launched with the following command:

 ```bash
 python app.py
 ```

Here is a demo GIF for the Gradio interface (accelerated).

<img src="media/gradio_office.gif" style="zoom: 66%;" />


## Evaluation on the Replica dataset

1. Download the Replica dataset generated by the authors of iMAP:
```bash
cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm -rf Replica.zip
```

2. Obtain the GT pointmaps and valid masks for each frame by running the following command:
```bash
python evaluation/process_gt.py
```
The processed GT will be saved at `./results/gt/replica`.

3. Evaluate the reconstruction on the Replica dataset with the following command:

```bash
bash ./scripts/eval_replica.sh
```

Both the numerical results and the error heatmaps will be saved in the directory `./results/TEST_NAME/eval/`.

> [!NOTE]
> Different versions of CUDA, PyTorch, and xformers can lead to slight variations in the predicted point cloud. These differences may be amplified during the alignment process in evaluation. Consequently, the numerical results you obtain might differ from those reported in the paper. However, the average values should remain approximately the same.

## Training

### Datasets

We use ScanNet++, Aria Synthetic Environments and Co3Dv2 to train our models. For data downloading and pre-processing, please refer to [here](./docs/data_preprocess.md). 

### Pretrained weights

```bash
# download the pretrained weights from DUSt3R
mkdir checkpoints 
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/
```

### Start training

```bash
# train the Image-to-Points model and the retrieval module
bash ./scripts/train_i2p.sh
# train the Local-to-World model
bash ./scripts/train_l2w.sh
```
> [!NOTE]
> They are not strictly equivalent to what was used to train SLAM3R, but they should be close enough.

## Real-time Processing

### Configuration

#### Environment Variables
- `RABBITMQ_HOST`: Message queue host
- `ENABLE_MESH_GENERATION`: Enable Open3D mesh generation
- `SAVE_SEGMENTS`: Save point clouds/trajectories per segment
- `SEGMENT_OUTPUT_DIR`: Directory for segment outputs

#### Key Parameters
- `configs/camera_intrinsics.yaml`: Camera calibration
- `configs/wild.yaml`: Reconstruction parameters
- Max points: 2M (configurable in SpatialPointCloudBuffer)
- Keyframe stride: Adaptive (1-10 based on motion)

## Performance Optimization

### Current Bottlenecks
- Point cloud downsampling: 47% of CPU time
- Python GIL limitations with ThreadPoolExecutor
- Synchronous mesh generation blocking main thread
- Inefficient data structures (numpy ↔ list conversions)

### Optimization Opportunities
- Remove downsampling (mesh generation already reduces volume)
- Increase RabbitMQ prefetch_count (currently 1)
- Fix INFERENCE_WINDOW_BATCH for proper GPU batching
- Use process pool for true async mesh generation
- Implement adaptive mesh generation based on camera motion

### Performance Tips
- Batch frame processing when possible
- Use msgpack for RabbitMQ serialization
- Offload heavy computation to separate processes
- Monitor memory usage (point cloud accumulation)

## Citation

If you find our work helpful in your research, please consider citing: 
```
@article{slam3r,
  title={SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos},
  author={Liu, Yuzheng and Dong, Siyan and Wang, Shuzhe and Yin, Yingda and Yang, Yanchao and Fan, Qingnan and Chen, Baoquan},
  journal={arXiv preprint arXiv:2412.09401},
  year={2024}
}
```


## Acknowledgments

Our implementation is based on several awesome repositories:

- [Croco](https://github.com/naver/croco)
- [DUSt3R](https://github.com/naver/dust3r)
- [NICER-SLAM](https://github.com/cvg/nicer-slam)
- [Spann3R](https://github.com/HengyiWang/spann3r)

We thank the respective authors for open-sourcing their code.