# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Android application for controlling DJI drones via WebSocket communication. The app integrates DJI Mobile SDK v5 and provides real-time drone control, video streaming, and sensor data transmission capabilities.

## Essential Build Commands

```bash
# Clean build artifacts
./gradlew clean

# Build debug APK
./gradlew assembleDebug

# Build and install on connected device
./gradlew installDebug

# Run unit tests
./gradlew test

# Run lint checks
./gradlew lint

# Display all available tasks
./gradlew tasks
```

## Architecture & Key Components

### Core Architecture
- **Language**: Kotlin (primary) with Java compatibility
- **Min SDK**: 31 (Android 12), Target SDK: 34 (Android 14)
- **Architecture**: arm64-v8a only (DJI SDK requirement)
- **WebSocket Server**: Default connection to 134.117.167.139

### Key Classes & Their Responsibilities

1. **WebSocket Communication** (app/src/main/java/com/tomscompany/websockettest/websocket/)
   - `WebsocketContainer`: Singleton managing WebSocket connections with auto-reconnection
   - `WebsocketMessageHandler`: Processes incoming messages

2. **DJI Integration** (app/src/main/java/com/tomscompany/websockettest/)
   - `DJIAircraftApplication`: Main application class extending DJIApplication
   - `MSDKManagerVM`: Manages SDK initialization
   - `BasicAircraftControlVM`: Basic flight operations
   - `VirtualStickVM`: Manual flight control
   - `LiveStreamVM`: Video streaming with frame skipping

3. **UI Components**
   - `FirstFragment`: Main UI with virtual stick controls
   - `OnScreenJoystick`: Custom joystick component

### Module Structure
- `:app` - Main application module
- `:android-sdk-v5-uxsdk` - DJI UI SDK module

## Development Workflow

### Testing a Single Feature
```bash
# Run specific test class
./gradlew :app:testDebugUnitTest --tests "com.tomscompany.websockettest.YourTestClass"

# Install and launch app
./gradlew installDebug && adb shell am start -n com.tomscompany.websockettest/.MainActivity
```

### WebSocket Development
The WebSocket system uses a singleton pattern with message queuing:
- Messages are queued when disconnected and sent upon reconnection
- Binary and text message types are supported
- Connection state is managed through `WebsocketContainer.connectionState`

### DJI SDK Integration Notes
- Requires arm64-v8a architecture
- API keys stored in gradle.properties (AIRCRAFT_API_KEY)
- SDK initialization happens in DJIAircraftApplication
- Virtual stick control requires specific permissions and flight mode

## Important Configuration

### API Keys (gradle.properties)
- `GMAP_API_KEY`: Google Maps API
- `AIRCRAFT_API_KEY`: DJI Aircraft SDK
- `MAPLIBRE_TOKEN`: MapLibre mapping

### Required Permissions
The app requires extensive permissions for drone control:
- Storage, Network, Location
- USB host/accessory
- Audio recording
- Phone state access

## Common Development Tasks

### Adding New WebSocket Messages
1. Define message type in WebsocketMessageHandler
2. Implement handler logic
3. Update WebsocketContainer if needed for special handling

### Implementing New Drone Controls
1. Add ViewModel in appropriate package
2. Register with MSDKManagerVM if SDK-dependent
3. Update FirstFragment for UI integration

### Debugging WebSocket Connection
- Check `WebsocketContainer.connectionState`
- Monitor reconnection attempts in logcat
- Default server: 134.117.167.139 (can be changed in WebsocketContainer)