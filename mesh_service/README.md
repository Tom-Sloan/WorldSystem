# Mesh Service v2.0

Real-time SLAM point cloud visualization with efficient voxel downsampling.

## Overview

Consumes SLAM3R keyframes via shared memory and visualizes in Rerun with Open3D voxel downsampling.

## Key Improvements

**Before (C++ v1.0)**:
- Memory: 5-10 GB after 5 minutes ❌
- Rerun FPS: <1 fps ❌
- Issue: Unlimited entity accumulation

**After (Python v2.0)**:
- Memory: ~50 MB constant ✅
- Rerun FPS: 60 fps ✅  
- Processing: 3-6 ms/keyframe ✅
- Single entity rendering ✅

## Running

```bash
# With SLAM3R
docker-compose --profile slam3r up

# View logs
docker logs -f mesh_service
```

## Configuration

Key environment variables:

```bash
VOXEL_SIZE=0.02          # 2cm voxels
MAX_POINTS=500000        # Memory cap
LOG_INTERVAL_MS=500      # Update rate
ENABLE_VIDEO=true        # Video streaming
```

## Architecture

```
RabbitMQ → Shared Memory Reader → Point Cloud Manager (Open3D)
                                         ↓
                                   Rerun Publisher
```

## Files

```
python/
├── main.py                    # Entry point
├── config.py                  # Environment config
├── io/
│   ├── shared_memory.py       # SHM reader
│   └── rabbitmq_consumer.py   # RabbitMQ
├── processing/
│   └── point_cloud_manager.py # Open3D voxel grid
└── visualization/
    ├── rerun_publisher.py     # Rerun logger
    └── blueprint_manager.py   # Layouts
```

## Troubleshooting

**High memory?** Reduce points:
```bash
VOXEL_SIZE=0.03
MAX_POINTS=300000
```

**Slow performance?** Increase intervals:
```bash
LOG_INTERVAL_MS=1000
```
