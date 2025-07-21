#!/usr/bin/env python3
"""
WebSocket server with H.264 video streaming support
"""

import asyncio
import struct
import logging
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from h264_handler import H264StreamHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize H.264 handler
h264_handler = H264StreamHandler()

# Track connected clients
connected_clients: Set[WebSocket] = set()
connected_phones: Dict[str, WebSocket] = {}

@app.get("/")
async def get():
    """Simple HTML page for testing"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>H.264 WebSocket Server</title>
    </head>
    <body>
        <h1>H.264 Video Streaming Server</h1>
        <p>Connect your Android app to ws://[server-ip]/ws/phone</p>
        <div id="status">Waiting for connections...</div>
        <div id="stats"></div>
        <script>
            // Simple status display
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(data => {
                    document.getElementById('status').innerText = 
                        `Connected phones: ${data.phones}`;
                    document.getElementById('stats').innerText = 
                        JSON.stringify(data.streams, null, 2);
                });
            }, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/status")
async def get_status():
    """Get server status"""
    streams = {}
    for client_id, info in h264_handler.streams.items():
        streams[client_id] = {
            "frames": info.total_frames,
            "keyframes": info.keyframes,
            "has_sps": info.sps is not None,
            "has_pps": info.pps is not None
        }
    
    return {
        "phones": len(connected_phones),
        "streams": streams
    }

@app.websocket("/ws/phone")
async def websocket_phone_endpoint(websocket: WebSocket):
    """Handle phone WebSocket connections"""
    await websocket.accept()
    client_id = str(id(websocket))
    connected_phones[client_id] = websocket
    logger.info(f"Phone connected: {client_id}")
    
    try:
        while True:
            # Receive data from phone
            data = await websocket.receive()
            
            if "bytes" in data:
                frame_bytes = data["bytes"]
                
                # Check if it's H.264 data by looking at packet structure
                if len(frame_bytes) > 4:
                    # Check for our packet header
                    total_size = struct.unpack('>I', frame_bytes[:4])[0]
                    
                    if total_size == len(frame_bytes):
                        # This is our H.264 packet format
                        decoded_frame = await h264_handler.handle_video_packet(
                            client_id, frame_bytes
                        )
                        
                        if decoded_frame is not None:
                            # Process decoded frame
                            await process_video_frame(client_id, decoded_frame)
                            
                            # Get stream info
                            info = h264_handler.get_stream_info(client_id)
                            if info and info.total_frames % 30 == 0:
                                logger.info(
                                    f"Client {client_id}: {info.total_frames} frames, "
                                    f"{info.keyframes} keyframes"
                                )
                    else:
                        # Legacy JPEG format (has 8-byte timestamp prefix)
                        if len(frame_bytes) > 8:
                            timestamp = struct.unpack('>Q', frame_bytes[:8])[0]
                            jpeg_data = frame_bytes[8:]
                            logger.debug(f"Received JPEG frame: {len(jpeg_data)} bytes")
                        
            elif "text" in data:
                # Handle text messages
                text_data = data["text"]
                logger.info(f"Received text from {client_id}: {text_data}")
                
                # Echo back for testing
                await websocket.send_text(f"Echo: {text_data}")
                
    except WebSocketDisconnect:
        logger.info(f"Phone disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error handling phone {client_id}: {e}")
    finally:
        # Clean up
        if client_id in connected_phones:
            del connected_phones[client_id]
        h264_handler.cleanup_stream(client_id)

async def process_video_frame(client_id: str, frame: any):
    """Process a decoded video frame"""
    # Here you can:
    # - Save the frame to disk
    # - Stream it to other clients
    # - Perform computer vision analysis
    # - etc.
    
    # For now, just log that we received it
    if frame is not None:
        height, width = frame.shape[:2]
        logger.debug(f"Decoded frame from {client_id}: {width}x{height}")

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info("H.264 WebSocket server starting...")
    logger.info("Make sure to install PyAV: pip install av")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down server...")
    # Close all connections
    for websocket in list(connected_phones.values()):
        await websocket.close()

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "websocket_server_h264:app",
        host="0.0.0.0",
        port=80,
        log_level="info",
        reload=True
    )