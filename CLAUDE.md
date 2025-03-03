# WorldSystem Project Guidelines
When a file becomes too long, split it into smaller files. When a function becomes too long, split it into smaller functions.

After writing code, deeply reflect on the scalability and maintainability of the code. Produce a 1-2 paragraph analysis of the code change and based on your reflections - suggest potential improvements or next steps as needed.

## Build Commands
- Full project: `docker-compose build`
- Single service: `docker-compose build --no-cache <service_name>`
- Run all services: `docker-compose up`

## Development Commands
- Website: `cd website && npm run dev`
- Reconstruction: `cd reconstruction && python main.py --cfg ./config/train.yaml`
- SLAM: `cd slam/demo && python run_rgbd.py PATH --vocab_file=./Vocabulary/ORBvoc.txt`

## Test Commands
- Reconstruction: `cd reconstruction && ./test.sh`
- Website: `cd website && npm run lint`

## Code Style Guidelines
- **Python**: Use PEP 8 conventions, 4-space indentation, and docstrings
- **JavaScript/React**: Follow ESLint configuration, organize imports alphabetically
- **Naming**: Use snake_case for Python variables/functions, camelCase for JavaScript
- **Error handling**: Use try/except in Python, catch and log errors in JavaScript
- **Comments**: Add docstrings to Python functions, JSDoc to JavaScript components

## Only modify the code in:
- `reconstruction/aaa/`
- `slam/aaa/`
- `website/`
- `server/`
- `storage/`
- `simulation/`
- `nginx/`
- `fantasy/`
- `docker/`
- `assets/`
- `Drone_Camera_Imu_Config/`
- `Drone_Calibration/`
- `docker-compose.yml`
- any Dockerfile
- `README.md`
- `prometheus.yml`

