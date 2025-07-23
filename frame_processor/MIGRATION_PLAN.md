# Frame Processor Migration Plan: RabbitMQ + WebSocket Unification

## Overview
This plan details the migration from having two separate implementations (RabbitMQ-based and WebSocket-based) to a single unified implementation that uses WebSocket for video input and RabbitMQ for result publishing.

## Current State Analysis

### main.py (RabbitMQ Version)
- **Class**: `FrameProcessorService`
- **Video Input**: RabbitMQ queue (`video_stream_exchange`)
- **Processing**: Async message handling with `process_stream_message()`
- **H.264 Decoding**: Uses `H264StreamDecoder` class
- **Publishing**: RabbitMQ via `RabbitMQPublisher`
- **Metrics**: Comprehensive Prometheus metrics
- **Performance**: Includes performance monitor dashboard

### websocket_frame_processor.py (WebSocket Version)
- **Class**: `WebSocketFrameProcessor` (inherits from `WebSocketVideoConsumer`)
- **Video Input**: WebSocket `/ws/video/consume`
- **Processing**: Synchronous `process_frame()` method
- **H.264 Decoding**: Built into base class
- **Publishing**: Same RabbitMQ publisher
- **Extras**: Rerun H.264 streaming test code

## Migration Strategy

### Phase 1: Preserve Existing Structure
Keep the `FrameProcessorService` class and its naming conventions, but modify it to:
1. Add WebSocket consumer capability alongside RabbitMQ
2. Replace video stream consumption from RabbitMQ with WebSocket
3. Keep RabbitMQ for:
   - Publishing results (`processed_frames_exchange`, `api_results_exchange`)
   - Receiving analysis mode updates (`analysis_mode_exchange`)
   - Other control messages

### Phase 2: Integration Approach
```python
class FrameProcessorService:
    def __init__(self):
        # Existing initialization...
        self.websocket_consumer = None  # New WebSocket consumer
        
    async def connect_websocket(self):
        """Connect to WebSocket for video streaming"""
        # Use WebSocketVideoConsumer as a component
        
    async def process_websocket_frame(self, frame, frame_number):
        """Process frame from WebSocket (replaces process_stream_message)"""
        # Reuse existing processing logic
```

### Phase 3: Specific Changes

#### 1. Modify FrameProcessorService.__init__()
- Add WebSocket configuration
- Keep all existing components

#### 2. Update connect() method
- Remove video_stream_exchange declaration
- Remove stream_queue declaration and binding
- Keep mode_queue for analysis updates
- Add WebSocket connection initialization

#### 3. Replace Video Consumption
- Remove: `await self.stream_queue.consume(self.process_stream_message)`
- Add: WebSocket consumer task
- Create adapter to feed WebSocket frames into existing pipeline

#### 4. Adapt Processing Flow
```
Old: RabbitMQ Message → H.264 Decoder → Frames → VideoProcessor
New: WebSocket → Base Class Decoder → Frames → VideoProcessor
```

#### 5. Keep Existing Features
- Performance monitoring dashboard
- Prometheus metrics
- NTP synchronization
- API integrations
- Result publishing via RabbitMQ

## Implementation Steps

### Step 1: Create WebSocket Adapter
```python
class WebSocketVideoAdapter(WebSocketVideoConsumer):
    """Adapter to integrate WebSocket consumer with existing pipeline"""
    def __init__(self, frame_processor_service, ws_url):
        super().__init__(ws_url, "FrameProcessor", frame_skip=1)
        self.service = frame_processor_service
        
    def process_frame(self, frame: np.ndarray, frame_number: int):
        """Bridge to existing processing pipeline"""
        # Convert to async and call service.process_frame()
```

### Step 2: Modify main.py
1. Import WebSocket consumer base class
2. Add WebSocket URL configuration
3. Create WebSocket adapter instance
4. Replace RabbitMQ video consumption with WebSocket task
5. Keep all other RabbitMQ functionality

### Step 3: Merge Docker Configuration
- Combine requirements from both Dockerfiles
- Add WebSocket dependencies (websockets, av)
- Keep all existing dependencies
- Single entry point: main.py

### Step 4: Update Configuration
- Add `WEBSOCKET_VIDEO_URL` environment variable
- Keep all existing RabbitMQ configuration
- Remove redundant video stream exchange config

### Step 5: Cleanup
- Remove websocket_frame_processor.py
- Remove Dockerfile.websocket
- Update documentation
- Remove test scripts specific to WebSocket version

## Benefits of This Approach

1. **Minimal Code Changes**: Preserves existing structure and naming
2. **Backward Compatible**: Keeps all RabbitMQ functionality except video input
3. **Clean Architecture**: WebSocket is just another input source
4. **Preserves Features**: All existing features remain intact
5. **Single Entry Point**: One main.py, one Dockerfile

## Risk Mitigation

1. **Backup**: Create main.py.backup before changes
2. **Testing**: Test WebSocket connection separately first
3. **Gradual Migration**: Can run both versions during transition
4. **Rollback Plan**: Keep backup files for quick reversion

## Validation Checklist

- [ ] WebSocket connection established
- [ ] H.264 frames decoded correctly
- [ ] Video processing pipeline works
- [ ] Results published to RabbitMQ
- [ ] Metrics and monitoring functional
- [ ] Performance dashboard operational
- [ ] API integrations working
- [ ] Docker container builds and runs

## Timeline

1. **Hour 1**: Backup and initial WebSocket adapter creation
2. **Hour 2**: Modify main.py to integrate WebSocket
3. **Hour 3**: Test and debug integration
4. **Hour 4**: Merge Docker files and cleanup
5. **Hour 5**: Documentation and final testing