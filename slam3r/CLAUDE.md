# SLAM3R Project Guidelines

## Build Commands
- Full build: `docker-compose build slam3r`
- Run service: `docker-compose up slam3r`
- Python environment: `conda create -n slam3r python=3.11 cmake=3.14.0`
- Install dependencies: `pip install -r SLAM3R_engine/requirements.txt`
- Optional dependencies: `pip install -r SLAM3R_engine/requirements_optional.txt`

## Demo/Evaluation Commands
- Run demo on Replica dataset: `bash SLAM3R_engine/scripts/demo_replica.sh`
- Run demo on wild data: `bash SLAM3R_engine/scripts/demo_wild.sh`
- Visualize reconstruction: `bash SLAM3R_engine/scripts/demo_vis_wild.sh`
- Gradio interface: `cd SLAM3R_engine && python app.py`
- Evaluate on Replica: `cd SLAM3R_engine && bash ./scripts/eval_replica.sh`
- Process ground truth: `cd SLAM3R_engine && python evaluation/process_gt.py`

## Training Commands
- Train I2P model: `cd SLAM3R_engine && bash ./scripts/train_i2p.sh`
- Train L2W model: `cd SLAM3R_engine && bash ./scripts/train_l2w.sh`

## Code Style Guidelines
- **Python**: Use PEP 8 conventions with 4-space indentation
- **Imports**: Group imports (standard library, third-party, local)
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Naming**: Use snake_case for variables/functions, CamelCase for classes
- **Error handling**: Use specific exceptions with context information
- **Logging**: Prefer the logging module over print statements