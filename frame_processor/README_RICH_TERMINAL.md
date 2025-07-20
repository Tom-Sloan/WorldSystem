# Running Frame Processor with Rich Terminal Display

The frame processor supports two display modes:
1. **Simple mode** (default in Docker): Periodic log messages
2. **Rich mode**: Beautiful, auto-updating terminal dashboard

## Quick Start - Rich Terminal in Docker

### Method 1: One-line command
```bash
ENABLE_RICH_TERMINAL=true docker-compose --profile frame_processor up frame_processor
```

### Method 2: Using docker run with the built image
First build:
```bash
docker-compose build frame_processor
```

Then run interactively:
```bash
docker run -it --rm \
  --runtime=nvidia \
  --network=host \
  -e ENABLE_RICH_TERMINAL=true \
  -e IS_DOCKER=false \
  -e RABBITMQ_URL=amqp://127.0.0.1:5672 \
  frame_processor:latest
```

### Method 3: Set in .env file
Add to `.env`:
```
ENABLE_RICH_TERMINAL=true
```

Then run normally:
```bash
docker-compose --profile frame_processor up frame_processor
```

## What You'll See

### Rich Terminal Mode
```
╭─────────────────────────────────────────────────────────────────╮
│      🎥 Frame Processor Performance Monitor  |  Runtime: 00:05:23 │
╰─────────────────────────────────────────────────────────────────╯

╭─ Overview ──────────╮  ╭─ Operation Timings ─────────────────────────╮
│ 📊 FPS: 28.3       │  │ Operation              Count  Avg    Min    │
│ 🎯 Frames: 8,523   │  │ detection              8523   32.1ms 28.3ms │
│ 👥 Active Tracks: 5 │  │ tracking               8523   5.2ms  3.1ms  │
│ 🔍 Avg Detections: 3│  │ visualization          8523   12.3ms 8.5ms  │
...
```

### Simple Mode (Default in Docker)
```
[PERF] FPS: 28.3 | Frames: 100 | Tracks: 5 | Detections/frame: 3.2 | Memory: 512MB | GPU: 1248MB
[TIMING] Top operations: detection: 32.1ms, visualization: 12.3ms, tracking: 5.2ms
```

## Troubleshooting

If the rich terminal doesn't display properly:
1. Make sure your terminal supports ANSI escape codes
2. Check that Docker is running with TTY support
3. Try running outside Docker for best results
4. Ensure ENABLE_RICH_TERMINAL=true is set

## Running Outside Docker

For the best experience, run directly:
```bash
cd frame_processor
python main.py
```

This will automatically detect TTY and use the rich display.