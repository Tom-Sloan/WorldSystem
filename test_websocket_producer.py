#!/usr/bin/env python3
"""
Test WebSocket video producer - simulates Android app sending H.264 with custom headers
"""

import asyncio
import websockets
import struct
import time
import sys

async def send_test_video():
    """Send test H.264 data with custom headers to server"""
    ws_url = "ws://localhost:5001/ws/video"
    
    print(f"Connecting to {ws_url} as video producer...")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✓ Connected as video producer")
            
            # Example SPS NAL unit (from a real H.264 stream)
            sps_nal = bytes([0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f, 
                           0x96, 0x54, 0x05, 0x01, 0x7b, 0xcb, 0x37, 0x01, 
                           0x01, 0x01, 0x02])
            
            # Example PPS NAL unit
            pps_nal = bytes([0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x3c, 0x80])
            
            # Example IDR frame start (keyframe)
            idr_start = bytes([0x00, 0x00, 0x00, 0x01, 0x65])
            
            # Example non-IDR frame start
            non_idr_start = bytes([0x00, 0x00, 0x00, 0x01, 0x41])
            
            frame_number = 0
            
            # Send SPS with custom header
            print("\nSending SPS...")
            timestamp = int(time.time() * 1e9)  # nanoseconds
            header = struct.pack('>Q', timestamp)  # 8 bytes timestamp
            header += struct.pack('>I', frame_number)  # 4 bytes frame number
            header += struct.pack('B', 0x01)  # 1 byte flags (0x01 = SPS)
            packet = header + sps_nal
            await websocket.send(packet)
            print(f"  Sent SPS packet: {len(packet)} bytes (header: {len(header)}, NAL: {len(sps_nal)})")
            
            await asyncio.sleep(0.1)
            
            # Send PPS with custom header
            print("\nSending PPS...")
            timestamp = int(time.time() * 1e9)
            header = struct.pack('>Q', timestamp)
            header += struct.pack('>I', frame_number)
            header += struct.pack('B', 0x02)  # 0x02 = PPS
            packet = header + pps_nal
            await websocket.send(packet)
            print(f"  Sent PPS packet: {len(packet)} bytes")
            
            await asyncio.sleep(0.1)
            
            # Send some frames
            for i in range(20):
                frame_number += 1
                timestamp = int(time.time() * 1e9)
                
                # Every 5th frame is a keyframe
                is_keyframe = (i % 5 == 0)
                
                if is_keyframe:
                    print(f"\nSending keyframe {frame_number}...")
                    flags = 0x03  # KEYFRAME
                    nal_data = idr_start + bytes([0x88] * 1000)  # Dummy frame data
                else:
                    print(f"Sending frame {frame_number}...")
                    flags = 0x04  # FRAME
                    nal_data = non_idr_start + bytes([0x88] * 500)  # Dummy frame data
                
                # Build header (21 bytes for frames)
                header = struct.pack('>Q', timestamp)  # 8 bytes timestamp
                header += struct.pack('>I', frame_number)  # 4 bytes frame number
                header += struct.pack('B', flags)  # 1 byte flags
                header += struct.pack('>I', len(nal_data))  # 4 bytes data length
                header += struct.pack('>I', 0)  # 4 bytes reserved
                
                packet = header + nal_data
                await websocket.send(packet)
                print(f"  Sent {'keyframe' if is_keyframe else 'frame'} packet: {len(packet)} bytes")
                
                # Simulate 30 FPS
                await asyncio.sleep(1/30)
            
            print("\n✓ Test complete! Sent 20 frames")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    return True


async def check_consumers():
    """Check how many consumers are connected"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:5001/video/status")
            status = resp.json()
            print(f"\nCurrent video status:")
            print(f"  Active consumers: {status.get('websocket_consumers', 0)}")
            print(f"  Has SPS/PPS: {status.get('has_sps_pps_cached', False)}")
            return status
    except Exception as e:
        print(f"❌ Could not check status: {e}")
        return None


async def main():
    print("WebSocket Video Producer Test")
    print("=============================")
    print("This simulates what the Android app sends\n")
    
    # Check initial status
    await check_consumers()
    
    # Send test video
    await send_test_video()
    
    # Check final status
    await check_consumers()


if __name__ == "__main__":
    asyncio.run(main())