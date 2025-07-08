#!/usr/bin/env python3
"""
Test script to verify SLAM3R segment reset functionality.
This script monitors the SLAM3R logs and RabbitMQ messages to confirm:
1. Segment boundaries are detected
2. SLAM system is reset on segment change
3. Point cloud data is optionally saved
"""

import asyncio
import json
import aio_pika
import os
import sys
from datetime import datetime

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://127.0.0.1:5672")
SLAM3R_RECONSTRUCTION_VIS_EXCHANGE = "slam3r_reconstruction_vis_exchange"

async def monitor_segment_changes():
    """Monitor RabbitMQ for segment boundary messages."""
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    
    async with connection:
        channel = await connection.channel()
        
        # Declare the exchange
        exchange = await channel.declare_exchange(
            SLAM3R_RECONSTRUCTION_VIS_EXCHANGE, 
            aio_pika.ExchangeType.FANOUT, 
            durable=True
        )
        
        # Create a temporary queue
        queue = await channel.declare_queue(exclusive=True)
        await queue.bind(exchange)
        
        print(f"[{datetime.now()}] Monitoring for segment boundary messages...")
        print("Expected behavior:")
        print("1. When video segments change, you should see 'segment_boundary' messages")
        print("2. Check SLAM3R logs for reset messages")
        print("3. If SLAM3R_SAVE_SEGMENT_POINTCLOUDS=true, check output directory\n")
        
        segment_count = 0
        
        # Process messages
        async for message in queue:
            try:
                data = json.loads(message.body.decode())
                
                if data.get("type") == "segment_boundary":
                    segment_count += 1
                    print(f"\n[{datetime.now()}] SEGMENT BOUNDARY DETECTED #{segment_count}")
                    print(f"  Previous segment: {data.get('previous_segment')}")
                    print(f"  New segment: {data.get('new_segment')}")
                    print(f"  Timestamp: {data.get('timestamp_ns')}")
                    
                    # Check if point cloud files were saved
                    if os.getenv("SLAM3R_SAVE_SEGMENT_POINTCLOUDS", "false").lower() == "true":
                        output_dir = os.getenv("SLAM3R_SEGMENT_OUTPUT_DIR", "/tmp/slam3r_segments")
                        print(f"\n  Check for saved files in: {output_dir}")
                        
                        if os.path.exists(output_dir):
                            files = list(os.listdir(output_dir))
                            if files:
                                print(f"  Found {len(files)} files:")
                                for f in sorted(files)[-6:]:  # Show last 6 files
                                    print(f"    - {f}")
                
            except Exception as e:
                print(f"Error processing message: {e}")
            
            await message.ack()

if __name__ == "__main__":
    print("=== SLAM3R Segment Reset Test Monitor ===")
    print(f"Connecting to RabbitMQ at: {RABBITMQ_URL}")
    print("\nTo test:")
    print("1. Ensure SLAM3R is running with segment video data")
    print("2. Watch for segment boundary detections")
    print("3. Check SLAM3R container logs: docker logs slam3r")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        asyncio.run(monitor_segment_changes())
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)