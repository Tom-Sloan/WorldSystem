Here are the essential documentation links to review before implementing the H.264 video streaming solution:

## DJI SDK Documentation

### DJI Mobile SDK V5

-   **Camera Stream Manager**: https://developer.dji.com/doc/mobile-sdk-tutorial/en/camera/camera-stream.html
-   **ICameraStreamManager Interface**: https://developer.dji.com/api-reference-v5/android-api/Components/ICameraStreamManager.html
-   **Video Feed and Decoding**: https://developer.dji.com/doc/mobile-sdk-tutorial/en/camera/video-feed-decoding.html
-   **Camera Key Reference**: https://developer.dji.com/api-reference-v5/android-api/Components/KeyManager/CameraKey.html
    https://developer.dji.com/api-reference-v5/android-api/Components/SDKManager/DJISDKManager.htm
    https://developer.dji.com/api-reference-v5/android-api/Components/IMediaDataCenter/ICameraStreamManager.html

## Android Media APIs

### MediaCodec Documentation

-   **MediaCodec Overview**: https://developer.android.com/reference/android/media/MediaCodec
-   **MediaCodec Guide**: https://developer.android.com/guide/topics/media/mediacodec
-   **Surface Input for Encoding**: https://source.android.com/docs/core/media/mediacodec#surface-input
-   **Low Latency Video**: https://source.android.com/docs/core/media/low-latency

### MediaFormat Documentation

-   **MediaFormat Reference**: https://developer.android.com/reference/android/media/MediaFormat
-   **Supported Media Formats**: https://developer.android.com/guide/topics/media/media-formats

## H.264/AVC Specifications

### Core H.264 Resources

-   **H.264 NAL Units Explained**: https://yumichan.net/video-processing/video-compression/introduction-to-h264-nal-unit/
-   **H.264 Bitstream Guide**: https://www.cardinalpeak.com/blog/the-h-264-sequence-parameter-set
-   **ITU-T H.264 Specification**: https://www.itu.int/rec/T-REC-H.264

### Streaming Protocols

-   **RTP Payload Format for H.264**: https://datatracker.ietf.org/doc/html/rfc6184
-   **H.264 over WebSocket**: https://github.com/kclyu/rpi-webrtc-streamer/blob/master/docs/websocket_signaling_protocol.md

## WebSocket Implementation

### WebSocket Binary Data

-   **OkHttp WebSocket**: https://square.github.io/okhttp/4.x/okhttp/okhttp3/-web-socket/
-   **Binary WebSocket Frames**: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_servers#exchanging_data_frames
-   **FastAPI WebSockets**: https://fastapi.tiangolo.com/advanced/websockets/

## Server-Side Video Processing

### Python Video Libraries

-   **PyAV Documentation**: https://pyav.org/docs/stable/
-   **OpenCV Video I/O**: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
-   **aiortc (WebRTC)**: https://aiortc.readthedocs.io/en/latest/

### Hardware Decoding

-   **NVIDIA Video Codec SDK**: https://developer.nvidia.com/nvidia-video-codec-sdk
-   **PyNvCodec Documentation**: https://github.com/NVIDIA/VideoProcessingFramework
-   **FFmpeg Hardware Acceleration**: https://trac.ffmpeg.org/wiki/HWAccelIntro

## Network and Performance

### Bandwidth and Latency

-   **Video Bitrate Guidelines**: https://support.google.com/youtube/answer/2853702
-   **WebRTC Stats API**: https://www.w3.org/TR/webrtc-stats/
-   **Network Adaptive Streaming**: https://developer.apple.com/documentation/http_live_streaming/hls_authoring_specification_for_apple_devices

### Quality of Service

-   **DSCP for Video**: https://www.cisco.com/c/en/us/td/docs/solutions/Enterprise/WAN_and_MAN/QoS_SRND/QoS-SRND-Book/QoSIntro.html
-   **Jitter Buffer Implementation**: https://webrtchacks.com/jitter-buffer-primer/

## Debugging and Analysis Tools

### Video Analysis

-   **FFprobe Documentation**: https://ffmpeg.org/ffprobe.html
-   **MediaInfo**: https://mediaarea.net/en/MediaInfo/Support/SDK
-   **H.264 Bitstream Analyzer**: https://github.com/aizvorski/h264bitstream

### Network Analysis

-   **Wireshark Video Analysis**: https://wiki.wireshark.org/H264
-   **Chrome WebRTC Internals**: chrome://webrtc-internals/

## Best Practices and Patterns

### Streaming Architecture

-   **Live Streaming Architecture**: https://aws.amazon.com/blogs/media/part-1-back-to-basics-gops-explained/
-   **Adaptive Bitrate Streaming**: https://bitmovin.com/adaptive-streaming/
-   **Low Latency Best Practices**: https://www.wowza.com/blog/low-latency-streaming

### Error Handling

-   **MediaCodec Error Handling**: https://source.android.com/docs/core/media/mediacodec#error-handling
-   **WebSocket Reconnection**: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/readyState

## Implementation Examples

### Reference Implementations

-   **Android Camera2 Video**: https://github.com/android/camera-samples/tree/main/Camera2Video
-   **WebRTC Android Sample**: https://github.com/webrtc/samples/tree/gh-pages/src/content/peerconnection/pc1
-   **DJI SDK Samples**: https://github.com/dji-sdk/Mobile-SDK-Android-V5

### Video Streaming Projects

-   **Live555 Streaming Media**: http://www.live555.com/liveMedia/
-   **GStreamer WebRTC Demo**: https://github.com/centricular/gstwebrtc-demos

## Important Considerations

Before implementing, pay special attention to:

1. **DJI SDK Limitations**: Review the specific capabilities of your drone model in the DJI documentation
2. **MediaCodec Profiles**: Ensure H.264 baseline profile for maximum compatibility
3. **NAL Unit Handling**: Understanding NAL unit types is crucial for proper stream handling
4. **Network Conditions**: Implement proper error handling for packet loss and jitter
5. **Licensing**: H.264 has patent implications - review for commercial use

These resources will give you a comprehensive understanding of all the components involved in implementing H.264 video streaming from the DJI drone to your server.
