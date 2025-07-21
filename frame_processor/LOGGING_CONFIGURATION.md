# Frame Processor Logging Configuration

This document explains how logging is configured in the frame_processor service and how to control various logging outputs.

## Overview

The frame processor uses Python's standard logging module with custom formatters and filters to provide clean, informative logs while suppressing noise from external libraries.

## Configuration Locations

1. **Main logging setup**: `core/utils.py` - `setup_logging()` function
2. **Performance monitoring**: `core/performance_monitor.py` - Uses Rich library for terminal UI
3. **Service initialization**: `main.py` - Calls `setup_logging()` at startup

## Key Features

### 1. Suppressed External Library Logging
The following libraries are set to WARNING level to reduce noise:
- `pika` (RabbitMQ client)
- `urllib3` (HTTP client)
- `google` (Google API client)
- `sam2` (SAM2 model)
- `detectron2` (Facebook AI framework)
- `fvcore` (Facebook core utilities)
- `iopath` (I/O path utilities)
- `cv2` (OpenCV)
- `PIL` (Python Imaging Library)
- `torchvision` (PyTorch vision)

### 2. Message Filtering
A custom `MessageFilter` class suppresses messages containing:
- "For numpy array image" (from SAM2 or other CV libraries)
- "Using cache found in" (from PyTorch)
- "Downloading: " (from model downloads)
- "to /root/.cache/torch" (from PyTorch caching)

### 3. Rich Terminal Integration
When `ENABLE_RICH_TERMINAL=true`:
- Performance dashboard is displayed using Rich library
- Console logging can be disabled to avoid conflicts
- Falls back to simple periodic logging in Docker/non-TTY environments

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FILE` | `false` | Enable file logging |
| `LOG_DIR` | `/app/logs` | Directory for log files |
| `ENABLE_RICH_TERMINAL` | `false` | Enable Rich terminal UI |
| `SUPPRESS_EXTERNAL_LOGS` | `true` | Filter out noisy external library messages |

## Usage Examples

### 1. Basic Configuration (Docker)
```bash
# Default - clean console output
docker-compose up frame_processor
```

### 2. Debug Mode with File Logging
```bash
# Verbose logging to files
LOG_LEVEL=DEBUG LOG_FILE=true docker-compose up frame_processor
```

### 3. Rich Terminal Mode (Interactive)
```bash
# Beautiful dashboard (requires TTY)
ENABLE_RICH_TERMINAL=true docker-compose run frame_processor
```

### 4. Minimal Logging
```bash
# Only warnings and errors
LOG_LEVEL=WARNING docker-compose up frame_processor
```

### 5. Detailed Analysis Mode (No Suppression)
```bash
# See ALL messages including from external libraries
SUPPRESS_EXTERNAL_LOGS=false LOG_LEVEL=DEBUG docker-compose up frame_processor
```

## Troubleshooting

### Problem: "INFO - For numpy array image..." messages
**Solution**: These are filtered by default when `SUPPRESS_EXTERNAL_LOGS=true` (default). 
- To see these messages for debugging: `SUPPRESS_EXTERNAL_LOGS=false`
- To add more patterns to filter, edit `MessageFilter.patterns_to_suppress` in `core/utils.py`

### Problem: Duplicate logging with Rich terminal
**Solution**: The console handler is automatically disabled when Rich is enabled. To override:
```python
setup_logging(log_level="INFO", disable_console_when_rich=False)
```

### Problem: Too much logging from SAM2
**Solution**: SAM2 and related libraries are set to WARNING level. To make it more restrictive:
```python
logging.getLogger("sam2").setLevel(logging.ERROR)
```

## Adding Custom Filters

To filter additional messages:

```python
# In core/utils.py
class MessageFilter(logging.Filter):
    def __init__(self, patterns_to_suppress: list = None):
        super().__init__()
        self.patterns_to_suppress = patterns_to_suppress or [
            "For numpy array image",
            "Your new pattern here",  # Add new patterns
        ]
```

## Performance Logging

The performance monitor provides two modes:

1. **Rich Terminal Mode**: Interactive dashboard with real-time metrics
2. **Simple Mode**: Periodic summary logs every 5 seconds

Example simple mode output:
```
[PERF] FPS: 15.2 | Frames: 1523 | Tracks: 3 | Detections/frame: 2.1 | Memory: 512MB | GPU: 2048MB
[TIMING] Top operations: frame_processing: 45.2ms, sam2_inference: 38.1ms, rabbitmq_publish: 2.3ms
```

## Best Practices

1. Use structured logging with `extra` parameter:
   ```python
   logger.info("Processing frame", extra={"frame_id": 123, "stream": "cam1"})
   ```

2. Use appropriate log levels:
   - DEBUG: Detailed information for debugging
   - INFO: General informational messages
   - WARNING: Warning messages
   - ERROR: Error messages that don't stop execution
   - CRITICAL: Critical errors that stop execution

3. Use the performance timer for tracking operations:
   ```python
   from core.performance_monitor import DetailedTimer
   
   with DetailedTimer("my_operation"):
       # Your code here
       pass
   ```