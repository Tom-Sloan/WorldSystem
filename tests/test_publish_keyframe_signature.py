#!/usr/bin/env python3
"""Test script to verify StreamingKeyframePublisher signature."""

import sys
import os
sys.path.append('/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine')

from shared_memory import StreamingKeyframePublisher
import inspect

# Get the signature of publish_keyframe
sig = inspect.signature(StreamingKeyframePublisher.publish_keyframe)
print(f"StreamingKeyframePublisher.publish_keyframe signature: {sig}")

# List all parameters
for param_name, param in sig.parameters.items():
    print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")