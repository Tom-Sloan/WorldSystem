# Consolidation Plan: Eliminating Duplication in WorldSystem

## Overview
This plan eliminates duplication in both Python code (RabbitMQ configs) and Docker Compose configuration.

## Phase 1: Shared Python Package

### 1.1 Package Structure
```
shared/
├── worldsystem_common/
│   ├── __init__.py
│   ├── rabbitmq_config.py
│   ├── constants.py
│   └── utils.py
└── setup.py
```

### 1.2 Update Service Dockerfiles
Each service's Dockerfile needs to install the shared package:

```dockerfile
# In server/Dockerfile, frame_processor/Dockerfile, etc.
# After creating conda environment, add:
COPY shared /tmp/shared
RUN conda run -n <env_name> pip install -e /tmp/shared
```

### 1.3 Update Service Imports
Replace local rabbitmq_config imports with:
```python
from worldsystem_common import EXCHANGES, ROUTING_KEYS, declare_exchanges
```

## Phase 2: Docker Compose Consolidation

### 2.1 Use YAML Anchors in docker-compose.yml
```yaml
version: '3.8'

# Define reusable configurations
x-common-env: &common-env
  RABBITMQ_URL: amqp://127.0.0.1:5672
  PYTHONUNBUFFERED: "1"
  TZ: ${TZ:-UTC}

x-gpu-env: &gpu-env
  <<: *common-env
  NVIDIA_VISIBLE_DEVICES: all
  CUDA_PATH: ${CUDA_PATH}
  LIBGL_ALWAYS_INDIRECT: ${LIBGL_ALWAYS_INDIRECT}
  DISPLAY: ${DISPLAY}
  HOME: ${HOME}

x-build-args: &build-args
  USERNAME: ${USERNAME}
  UID: ${UID}
  GID: ${GID}

x-volumes: &x11-volumes
  - ${X11SOCKET}:${X11SOCKET}
  - ${XAUTHORITY}:${XAUTHORITY}

x-gpu-config: &gpu-config
  runtime: nvidia
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:
  server:
    build:
      context: .
      dockerfile: ./server/Dockerfile
      args: *build-args
    environment: *common-env
    volumes: *x11-volumes
    # ... other configs

  frame_processor:
    <<: *gpu-config
    build:
      context: .
      dockerfile: ./frame_processor/Dockerfile
      args: *build-args
    environment:
      <<: *gpu-env
      INITIAL_ANALYSIS_MODE: none
      RERUN_VIEWER_ADDRESS: 0.0.0.0:9090
      RERUN_CONNECT_URL: rerun+http://127.0.0.1:9876/proxy
      RERUN_ENABLED: true
      METRICS_PORT: 8003
    volumes: *x11-volumes
    # ... other configs

  slam3r:
    <<: *gpu-config
    build:
      context: .
      dockerfile: ./slam3r/Dockerfile
      args: *build-args
    environment:
      <<: *gpu-env
      # SLAM3R specific vars...
    volumes: *x11-volumes
    # ... other configs
```

### 2.2 Alternative: Use .env File for Common Values
Create a `.env.common` file:
```bash
# Common environment variables
RABBITMQ_URL=amqp://127.0.0.1:5672
PYTHONUNBUFFERED=1
NVIDIA_VISIBLE_DEVICES=all
```

Then in docker-compose.yml:
```yaml
services:
  server:
    env_file:
      - .env
      - .env.common
```

## Phase 3: Implementation Steps

### Step 1: Create Shared Package
```bash
# Create package structure
mkdir -p shared/worldsystem_common
touch shared/worldsystem_common/__init__.py
cp server/rabbitmq_config.py shared/worldsystem_common/
echo "from setuptools import setup..." > shared/setup.py
```

### Step 2: Update Service Dockerfiles
For each service (server, frame_processor, storage, slam3r, simulation):
1. Add shared package copy: `COPY shared /tmp/shared`
2. Install package: `RUN pip install -e /tmp/shared`

### Step 3: Update Service Code
```bash
# For each service
sed -i 's/from rabbitmq_config/from worldsystem_common/g' service_file.py
rm service/rabbitmq_config.py
```

### Step 4: Test Each Service
```bash
docker-compose build <service_name>
docker-compose run --rm <service_name> python -c "from worldsystem_common import EXCHANGES; print(EXCHANGES)"
```

### Step 5: Update docker-compose.yml
Apply YAML anchors to eliminate duplication.

## Benefits

1. **Single Source of Truth**: RabbitMQ config in one place
2. **Easier Updates**: Change exchanges/routing keys in one file
3. **Reduced Docker Compose Size**: ~40% reduction using anchors
4. **Better Maintainability**: DRY principle applied consistently
5. **Easier Testing**: Shared package can have unit tests

## Additional Improvements

### 1. Shared Constants Module
```python
# shared/worldsystem_common/constants.py
DEFAULT_FRAME_RATE = 30
MAX_QUEUE_SIZE = 1000
DEFAULT_HEARTBEAT = 3600
NTP_SYNC_INTERVAL = 60
```

### 2. Shared Utils Module
```python
# shared/worldsystem_common/utils.py
def get_ntp_time():
    """Shared NTP time function"""
    pass

def setup_prometheus_metrics(port):
    """Common Prometheus setup"""
    pass
```

### 3. Environment Variable Validation
```python
# shared/worldsystem_common/config.py
from pydantic import BaseSettings

class ServiceConfig(BaseSettings):
    rabbitmq_url: str = "amqp://rabbitmq"
    metrics_port: int = 8000
    
    class Config:
        env_file = ".env"
```

## Monitoring Improvements

With consolidated configuration, add:
1. Exchange health checks
2. Queue depth monitoring
3. Message rate tracking
4. Automatic alerts for exchange issues

## Final Directory Structure
```
WorldSystem2/
├── shared/                    # NEW: Shared package
│   ├── worldsystem_common/
│   │   ├── __init__.py
│   │   ├── rabbitmq_config.py
│   │   ├── constants.py
│   │   ├── utils.py
│   │   └── config.py
│   └── setup.py
├── server/
│   ├── Dockerfile (updated)
│   └── main.py (uses shared)
├── frame_processor/
│   ├── Dockerfile (updated)
│   └── frame_processor.py (uses shared)
├── docker-compose.yml (with anchors)
└── docker-compose.override.yml (optional overrides)
```

This approach provides a clean, maintainable solution that scales well as the system grows.