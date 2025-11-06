"""
WebSocket H.264 video consumer for mesh service.

Connects to server's /ws/video/consume endpoint and decodes H.264 frames
using PyAV, following the same pattern as slam3r.
"""

import asyncio
import logging
import os
from typing import Optional, Callable
import numpy as np
import websockets
import av

logger = logging.getLogger(__name__)

# Max buffer size to prevent memory issues
MAX_BYTE_BUFFER_SIZE = 1 << 20  # 1MB


class WebSocketVideoConsumer:
    """
    WebSocket H.264 video consumer.

    Connects to server WebSocket endpoint, receives H.264 packets,
    decodes them to RGB frames using PyAV.
    """

    def __init__(self, ws_url: str):
        """
        Initialize WebSocket video consumer.

        Args:
            ws_url: WebSocket URL (e.g., ws://127.0.0.1:5001/ws/video/consume)
        """
        self.ws_url = ws_url
        self.codec: Optional[av.CodecContext] = None
        self.byte_buffer = bytearray()
        self.reconnect_delay = 5  # seconds
        self.running = False

        # PTS tracking for B-frame deduplication
        self.last_pts: Optional[int] = None
        self.pts_duplicates_skipped = 0

        # Callback for decoded frames
        self.frame_callback: Optional[Callable] = None

        # Statistics
        self.frames_decoded = 0
        self.bytes_received = 0

    def set_frame_callback(self, callback: Callable):
        """
        Set callback for decoded frames.

        Args:
            callback: Async function(frame: np.ndarray, timestamp_ns: int)
        """
        self.frame_callback = callback

    async def _process_h264_packet(self, data: bytes):
        """Process H.264 packet and decode to RGB frames."""
        try:
            # Append data to buffer
            self.byte_buffer.extend(data)
            self.bytes_received += len(data)

            # Log progress
            if self.bytes_received % (1024 * 1024) < len(data):  # Every MB
                logger.debug(f"Received {self.bytes_received // (1024*1024)} MB")

            # Prevent buffer overflow
            if len(self.byte_buffer) > MAX_BYTE_BUFFER_SIZE:
                logger.warning("Byte buffer overflow - clearing oldest data")
                self.byte_buffer = self.byte_buffer[-MAX_BYTE_BUFFER_SIZE:]

            # Parse buffer into valid packets
            packets = self.codec.parse(self.byte_buffer)
            if not packets:
                return

            # Process each packet
            for packet in packets:
                try:
                    frames = self.codec.decode(packet)
                    for frame in frames:
                        # Skip duplicate PTS frames (common with B-frames)
                        if hasattr(frame, 'pts') and frame.pts is not None:
                            if self.last_pts is not None and frame.pts == self.last_pts:
                                self.pts_duplicates_skipped += 1
                                if self.pts_duplicates_skipped % 100 == 0:
                                    logger.debug(f"Skipped {self.pts_duplicates_skipped} duplicate PTS frames")
                                continue
                            self.last_pts = frame.pts

                        # Convert to RGB numpy array
                        img_rgb = frame.to_ndarray(format='rgb24')
                        self.frames_decoded += 1

                        # Log first frame
                        if self.frames_decoded == 1:
                            logger.info(f"First frame decoded! Resolution: {frame.width}x{frame.height}")
                        elif self.frames_decoded % 100 == 0:
                            logger.debug(f"Decoded {self.frames_decoded} frames")

                        # Call callback if set
                        if self.frame_callback:
                            # Generate timestamp (TODO: extract from stream metadata)
                            import time
                            timestamp_ns = int(time.time() * 1e9)
                            await self.frame_callback(img_rgb, timestamp_ns)

                except Exception as e:
                    # Individual packet decode errors are normal at the beginning
                    # Log the actual exception type to determine correct catch
                    if "decode" in str(e).lower() or "invalid" in str(e).lower():
                        logger.debug(f"Skipping invalid packet ({type(e).__module__}.{type(e).__name__}): {e}")
                    else:
                        raise  # Re-raise unexpected exceptions

            # Update buffer (remove parsed data)
            consumed = sum(p.size for p in packets)
            self.byte_buffer = self.byte_buffer[consumed:]

            # Flush decoder for any remaining frames
            # Note: Only attempt flush if we successfully decoded packets
            # EOF errors during active streaming are expected and should not be logged as errors
            try:
                flushed_frames = self.codec.decode()  # Empty packet flushes
                for frame in flushed_frames:
                    # Skip duplicate PTS frames
                    if hasattr(frame, 'pts') and frame.pts is not None:
                        if self.last_pts is not None and frame.pts == self.last_pts:
                            self.pts_duplicates_skipped += 1
                            continue
                        self.last_pts = frame.pts

                    img_rgb = frame.to_ndarray(format='rgb24')
                    self.frames_decoded += 1

                    if self.frame_callback:
                        import time
                        timestamp_ns = int(time.time() * 1e9)
                        await self.frame_callback(img_rgb, timestamp_ns)

            except av.error.EOFError:
                # EOF during flush is expected when decoder buffer is empty during active streaming
                # This is NOT an error - it just means no buffered frames are available
                logger.debug("Decoder flush: no buffered frames available (EOF)")
            except Exception as e:
                # Log other flush exceptions at debug level (e.g., invalid data at boundaries)
                logger.debug(f"Flush decode exception ({type(e).__module__}.{type(e).__name__}): {e}")

        except av.error.InvalidDataError:
            # This is normal for partial packets, skip
            pass
        except Exception as e:
            logger.error(f"Error processing H.264 packet: {e}")

    async def run(self):
        """Main WebSocket consumer loop."""
        self.running = True
        logger.info(f"Starting WebSocket video consumer, connecting to: {self.ws_url}")

        while self.running:
            try:
                # Connect to WebSocket
                async with websockets.connect(
                    self.ws_url,
                    ping_timeout=30,
                    close_timeout=10
                ) as websocket:
                    logger.info(f"✅ Connected to WebSocket: {self.ws_url}")

                    # Initialize H.264 decoder
                    self.codec = av.CodecContext.create('h264', 'r')
                    self.codec.thread_type = 'AUTO'  # Enable multi-threading
                    self.codec.extradata = None  # Reset extradata for new stream

                    # Reset state for new stream
                    self.last_pts = None
                    self.pts_duplicates_skipped = 0
                    self.byte_buffer = bytearray()

                    # Process incoming messages
                    async for message in websocket:
                        if not self.running:
                            break

                        if isinstance(message, bytes):
                            await self._process_h264_packet(message)
                        else:
                            logger.debug(f"Non-binary WebSocket message received: {message}")

            except websockets.exceptions.ConnectionClosed:
                if self.running:
                    logger.warning("WebSocket connection closed. Reconnecting...")
                else:
                    break
            except Exception as e:
                if self.running:
                    logger.error(f"WebSocket error: {e}. Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
            finally:
                # Clean up codec
                if self.codec:
                    self.codec = None

        logger.info("WebSocket video consumer stopped")

    async def stop(self):
        """Stop the WebSocket consumer."""
        self.running = False
        logger.info("Stopping WebSocket video consumer...")

    def get_stats(self) -> dict:
        """Get consumer statistics."""
        return {
            'frames_decoded': self.frames_decoded,
            'bytes_received': self.bytes_received,
            'pts_duplicates_skipped': self.pts_duplicates_skipped,
            'buffer_size': len(self.byte_buffer)
        }
