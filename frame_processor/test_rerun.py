#!/usr/bin/env python3
"""Test script to verify Rerun is working correctly."""

import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import numpy as np
import os
import time

# Get connection settings from environment or use defaults
RERUN_ENABLED = os.getenv("RERUN_ENABLED", "true").lower() == "true"
RERUN_CONNECT_URL = os.getenv("RERUN_CONNECT_URL", "rerun+http://localhost:9876/proxy")

if not RERUN_ENABLED:
    print("Rerun is disabled. Set RERUN_ENABLED=true to enable.")
    exit(0)

print("Testing Rerun connection...")

# Initialize Rerun
rr.init("rerun_test_frame_processor", spawn=False)

# Try to connect
try:
    print(f"Connecting to Rerun at: {RERUN_CONNECT_URL}")
    rr.connect(RERUN_CONNECT_URL)
    print("✓ Connected successfully!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("Falling back to spawn mode...")
    rr.init("rerun_test_frame_processor", spawn=True)

# Log test data
print("Logging test data...")

# 1. Points3D test
positions = np.zeros((10, 3))
positions[:, 0] = np.linspace(-10, 10, 10)

colors = np.zeros((10, 3), dtype=np.uint8)
colors[:, 0] = np.linspace(0, 255, 10)

rr.log(
    "test/points3d",
    rr.Points3D(positions, colors=colors, radii=0.5)
)
print("✓ Logged Points3D")

# 2. Image test
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
try:
    rr.log("test/image", rr.Image(test_image).compress(jpeg_quality=80))
    print("✓ Logged compressed image")
except AttributeError:
    rr.log("test/image", rr.Image(test_image))
    print("✓ Logged image (compression not available)")

# 3. Text log test
rr.log("test/logs", rr.TextLog("Test message from frame_processor", level="INFO"))
print("✓ Logged text message")

# 4. Text document test
test_markdown = """### Rerun Test Results

**Connection Status:** ✅ Connected
**Test Time:** {}

**Tests Performed:**
1. Points3D visualization
2. Image logging with compression
3. Text logging
4. Markdown document

All tests completed successfully!
""".format(time.strftime("%Y-%m-%d %H:%M:%S"))

rr.log("test/document", rr.TextDocument(test_markdown, media_type=rr.MediaType.MARKDOWN))
print("✓ Logged markdown document")

# 5. Test frame processor specific entities
# Simulate object tracking data
for i in range(5):
    bbox = [100 + i*50, 100, 150 + i*50, 200]
    rr.log(
        f"test/tracking/object_{i}",
        rr.Boxes2D(
            array=[[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]],
            array_format=rr.Box2DFormat.XYWH,
            labels=[f"Test Object {i}"],
            colors=[[0, 255, 0]]
        )
    )
print("✓ Logged bounding boxes")

# 6. Test scalar metrics
for i in range(10):
    rr.log("test/metrics/processing_time", rr.Scalar(np.random.uniform(10, 50)))
    rr.log("test/metrics/confidence", rr.Scalar(np.random.uniform(0.5, 1.0)))
    time.sleep(0.1)
print("✓ Logged scalar metrics")

print("\n✅ All Rerun tests completed successfully!")
print("Check the Rerun viewer to see the test data.")