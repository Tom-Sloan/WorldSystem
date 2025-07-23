#!/usr/bin/env python3
"""
Test WebSocket video streaming
"""

import asyncio
import websockets
import sys
import time
import av

async def test_websocket_consumer():
    """Test connecting to WebSocket video stream and receiving frames"""
    ws_url = "ws://localhost:5001/ws/video/consume"
    
    print(f"Connecting to {ws_url}...")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✓ Connected to WebSocket stream")
            
            # H.264 decoder
            codec = av.CodecContext.create('h264', 'r')
            
            frame_count = 0
            start_time = time.time()
            
            while frame_count < 100:  # Receive 100 frames
                try:
                    # Receive H.264 data
                    h264_data = await websocket.recv()
                    
                    if isinstance(h264_data, bytes):
                        print(f"Received H.264 NAL unit: {len(h264_data)} bytes")
                        
                        # Check NAL unit type
                        if len(h264_data) > 4:
                            nal_type = h264_data[4] & 0x1F
                            if nal_type == 7:
                                print("  -> SPS (Sequence Parameter Set)")
                            elif nal_type == 8:
                                print("  -> PPS (Picture Parameter Set)")
                            elif nal_type == 5:
                                print("  -> IDR frame (keyframe)")
                            elif nal_type == 1:
                                print("  -> Non-IDR frame")
                        
                        # Try to decode
                        try:
                            packet = av.Packet(h264_data)
                            frames = codec.decode(packet)
                            
                            for frame in frames:
                                frame_count += 1
                                print(f"✓ Decoded frame {frame_count}: {frame.width}x{frame.height}")
                                
                        except av.AVError as e:
                            print(f"  Decode error (normal during init): {e}")
                            
                except Exception as e:
                    print(f"Error receiving frame: {e}")
                    break
                    
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"\nTest complete!")
            print(f"Received {frame_count} frames in {elapsed:.1f} seconds")
            print(f"Average FPS: {fps:.1f}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
        
    return True


async def test_websocket_broadcasting():
    """Test multiple consumers connecting simultaneously"""
    ws_url = "ws://localhost:5001/ws/video/consume"
    
    print(f"\nTesting multiple consumers...")
    
    async def consumer(consumer_id):
        try:
            async with websockets.connect(ws_url) as websocket:
                print(f"  Consumer {consumer_id} connected")
                
                # Receive a few frames
                for i in range(5):
                    data = await websocket.recv()
                    print(f"  Consumer {consumer_id} received frame {i+1}: {len(data)} bytes")
                    
                print(f"  Consumer {consumer_id} disconnecting")
                
        except Exception as e:
            print(f"  Consumer {consumer_id} error: {e}")
    
    # Create 3 consumers
    tasks = [consumer(i) for i in range(3)]
    await asyncio.gather(*tasks)
    
    print("Multi-consumer test complete!")


async def main():
    print("WebSocket Video Streaming Test")
    print("==============================")
    
    # First check if server is running
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:5001/video/status")
            status = resp.json()
            print(f"Server status: {status}")
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        print("Make sure the server is running: docker-compose up server")
        return
    
    # Test 1: Basic consumer
    print("\nTest 1: Basic WebSocket consumer")
    print("-" * 40)
    await test_websocket_consumer()
    
    # Test 2: Multiple consumers
    print("\nTest 2: Multiple simultaneous consumers")
    print("-" * 40)
    await test_websocket_broadcasting()


if __name__ == "__main__":
    # Check dependencies
    try:
        import websockets
        import av
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install websockets av")
        sys.exit(1)
        
    asyncio.run(main())