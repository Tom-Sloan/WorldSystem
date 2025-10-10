package com.tomscompany.websockettest

import android.util.Log
import dji.v5.manager.aircraft.virtualstick.Stick
import org.json.JSONException
import org.json.JSONObject
import dji.v5.common.callback.CommonCallbacks
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.v5.common.error.IDJIError

/**
 * This file was created to handle incoming WebSocket messages and translate them into drone commands.
 * Updated to implement a control scheme using:
 * - WASD keys for movement
 * - Arrow keys for rotation and altitude
 * - P key for camera streaming toggle
 * - T key for takeoff/land toggle
 */
class WebsocketMessageHandler(
    private val mainActivity: MainActivity,
    private val virtualStickVM: VirtualStickVM,
    private val liveStreamVM: LiveStreamVM,
    private val basicAircraftControlVM: BasicAircraftControlVM
) {

    private val TAG = "WebsocketMessageHandler"

    // Handle incoming WebSocket messages
    fun processMessage(message: String) {
        try {
            val jsonObject = JSONObject(message)
            val messageType = jsonObject.optString("type", "")
            
            when (messageType) {
                "movement" -> handleMovementCommand(jsonObject)
                "rotation" -> handleRotationCommand(jsonObject)
                "camera" -> handleCameraCommand(jsonObject)
                "flightmode" -> handleFlightModeCommand(jsonObject)
                else -> Log.d(TAG, "Unknown message type: $messageType")
            }
        } catch (e: JSONException) {
            Log.e(TAG, "Error parsing JSON message: ${e.message}")
            WebsocketContainer.sendError("Failed to parse message: ${e.message}")
        }
    }

    // Handle WASD movement commands
    private fun handleMovementCommand(json: JSONObject) {
        try {
            val x = json.getDouble("x") // -1.0 (left) to 1.0 (right)
            val y = json.getDouble("y") // -1.0 (backward) to 1.0 (forward)
            
            // Convert from -1.0 to 1.0 scale to the drone stick position scale
            val horizontalPosition = (x * Stick.MAX_STICK_POSITION_ABS).toInt()
            val verticalPosition = (y * Stick.MAX_STICK_POSITION_ABS).toInt()
            
            // Set the left stick position for movement
            virtualStickVM.setLeftPosition(horizontalPosition, verticalPosition)
            
            // Log and send confirmation
            val message = "Movement command received: x=$x, y=$y"
            Log.d(TAG, message)
            WebsocketContainer.sendStatus(message)
        } catch (e: Exception) {
            Log.e(TAG, "Error handling movement command: ${e.message}")
            WebsocketContainer.sendError("Failed to process movement command", e.message)
        }
    }

    // Handle arrow key commands for rotation and altitude
    private fun handleRotationCommand(json: JSONObject) {
        try {
            val yaw = json.getDouble("yaw") // -1.0 (rotate left) to 1.0 (rotate right)
            val z = json.getDouble("z")     // -1.0 (down) to 1.0 (up)
            
            // Convert from -1.0 to 1.0 scale to the drone stick position scale
            val horizontalPosition = (yaw * Stick.MAX_STICK_POSITION_ABS).toInt()
            val verticalPosition = (z * Stick.MAX_STICK_POSITION_ABS).toInt()
            
            // Set the right stick position for yaw rotation and altitude
            virtualStickVM.setRightPosition(horizontalPosition, verticalPosition)
            
            // Log and send confirmation
            val message = "Rotation command received: yaw=$yaw, z=$z"
            Log.d(TAG, message)
            WebsocketContainer.sendStatus(message)
        } catch (e: Exception) {
            Log.e(TAG, "Error handling rotation command: ${e.message}")
            WebsocketContainer.sendError("Failed to process rotation command", e.message)
        }
    }

    // Handle camera toggle commands
    private fun handleCameraCommand(json: JSONObject) {
        try {
            val action = json.getString("action")
            
            when (action) {
                "toggle" -> toggleCameraStreaming()
                "on" -> startCameraStreaming()
                "off" -> stopCameraStreaming()
                else -> {
                    Log.d(TAG, "Unknown camera action: $action")
                    WebsocketContainer.sendError("Unknown camera action: $action")
                }
            }
            
            // Log and send confirmation
            val message = "Camera command received: action=$action"
            Log.d(TAG, message)
            WebsocketContainer.sendStatus(message)
        } catch (e: Exception) {
            Log.e(TAG, "Error handling camera command: ${e.message}")
            WebsocketContainer.sendError("Failed to process camera command", e.message)
        }
    }

    // Handle takeoff/land toggle commands
    private fun handleFlightModeCommand(json: JSONObject) {
        try {
            val action = json.getString("action")
            
            when (action) {
                "takeoff" -> executeOnEnableVirtualStick { executeTakeoff() }
                "land" -> executeOnEnableVirtualStick { executeLanding() }
                else -> {
                    Log.d(TAG, "Unknown flight mode action: $action")
                    WebsocketContainer.sendError("Unknown flight mode action: $action")
                }
            }
            
            // Log and send confirmation
            val message = "Flight mode command received: action=$action"
            Log.d(TAG, message)
            WebsocketContainer.sendStatus(message)
        } catch (e: Exception) {
            Log.e(TAG, "Error handling flight mode command: ${e.message}")
            WebsocketContainer.sendError("Failed to process flight mode command", e.message)
        }
    }

    // Execute takeoff command
    private fun executeTakeoff() {
        basicAircraftControlVM.startTakeOff(object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
            override fun onSuccess(t: EmptyMsg?) {
                val message = "Takeoff started successfully"
                Log.d(TAG, message)
                WebsocketContainer.sendStatus(message)
            }

            override fun onFailure(error: IDJIError) {
                val message = "Failed to start takeoff: ${error.description()}"
                Log.e(TAG, message)
                WebsocketContainer.sendError(message)
            }
        })
    }

    // Execute landing command
    private fun executeLanding() {
        basicAircraftControlVM.startLanding(object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
            override fun onSuccess(t: EmptyMsg?) {
                val message = "Landing started successfully"
                Log.d(TAG, message)
                WebsocketContainer.sendStatus(message)
            }

            override fun onFailure(error: IDJIError) {
                val message = "Failed to start landing: ${error.description()}"
                Log.e(TAG, message)
                WebsocketContainer.sendError(message)
            }
        })
    }

    // Helper function to execute an action after ensuring virtual stick is enabled
    private fun executeOnEnableVirtualStick(action: () -> Unit) {
        if (virtualStickVM.isVirtualStickEnabled.value == true) {
            action()
        } else {
            Log.d(TAG, "Virtual Stick not enabled. Enabling...")
            WebsocketContainer.sendStatus("Virtual Stick not enabled. Enabling...")
            virtualStickVM.enableVirtualStick { success, error ->
                if (success) {
                    Log.d(TAG, "Virtual Stick enabled successfully")
                    WebsocketContainer.sendStatus("Virtual Stick enabled successfully")
                    action()
                } else {
                    Log.e(TAG, "Failed to enable Virtual Stick: $error")
                    WebsocketContainer.sendError("Failed to enable Virtual Stick", error)
                }
            }
        }
    }

    // Helper methods for camera streaming
    private fun toggleCameraStreaming() {
        if (liveStreamVM.isStreaming()) {
            stopCameraStreaming()
        } else {
            startCameraStreaming()
        }
    }

    private fun startCameraStreaming() {
        if (liveStreamVM.isStreaming()) {
            WebsocketContainer.sendStatus("Stream is already running")
            return
        }

        mainActivity.runOnUiThread {
            liveStreamVM.startFrameStreaming()
            WebsocketContainer.sendStatus("Stream started successfully")
        }
    }

    private fun stopCameraStreaming() {
        if (!liveStreamVM.isStreaming()) {
            WebsocketContainer.sendStatus("Stream is not running")
            return
        }

        mainActivity.runOnUiThread {
            liveStreamVM.stopFrameStreaming()
            WebsocketContainer.sendStatus("Stream stopped successfully")
        }
    }
} 