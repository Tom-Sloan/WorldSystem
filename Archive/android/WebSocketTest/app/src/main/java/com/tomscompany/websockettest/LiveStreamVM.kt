package com.tomscompany.websockettest

import androidx.lifecycle.ViewModel
import android.util.Log
import android.view.Surface
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.manager.interfaces.ICameraStreamManager
import android.util.Base64
import org.json.JSONObject
import dji.v5.manager.interfaces.ICameraStreamManager.CameraFrameListener
import dji.v5.manager.interfaces.ICameraStreamManager.FrameFormat
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.common.utils.CallbackUtils
import dji.v5.manager.datacenter.MediaDataCenter
import com.tomscompany.websockettest.data.ImuData
import java.io.FileWriter
import org.json.JSONArray

class LiveStreamVM : ViewModel() {
    private val TAG = "LiveStreamVM"
    private val cameraStreamManager: ICameraStreamManager = MediaDataCenter.getInstance().cameraStreamManager
    private var isFrameStreaming = false
    private var frameListener: CameraFrameListener? = null
    private var frameCounter = 0
    private val FRAME_SKIP = 2 // Send every 2nd frame
    private val COMPRESSION_QUALITY = 70 // JPEG compression quality (0-100)
    private var fragment: FirstFragment? = null
    private var isRecording = false
    private var recordingFile: java.io.FileWriter? = null
    private var videoDirectory: java.io.File? = null
    private var currentVideoFile: java.io.FileOutputStream? = null
    private var frameIndex = 0

    init {
        
    }

    fun isStreaming(): Boolean {
        return isFrameStreaming
    }

    fun setVideoSurface(surface: Surface?, width: Int, height: Int) {
        try {
            if (surface != null) {
                cameraStreamManager.putCameraStreamSurface(
                    ComponentIndexType.LEFT_OR_MAIN,
                    surface,
                    width,
                    height,
                    ICameraStreamManager.ScaleType.CENTER_INSIDE
                )
                Log.d(TAG, "Video surface set successfully")
            } else {
                Log.d(TAG, "Video surface removed successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error setting video surface: ${e.message}")
            WebsocketContainer.send("Error setting video surface: ${e.message}")
        }
    }

    fun startFrameStreaming() {
        if (!isFrameStreaming) {
            Log.d(TAG, "Starting frame streaming")
            isFrameStreaming = true
            frameCounter = 0
            setupFrameListener()
            Log.d(TAG, "Frame streaming to server started")
        } else {
            Log.d(TAG, "Frame streaming already active")
        }
    }

    fun stopFrameStreaming() {
        if (isFrameStreaming) {
            isFrameStreaming = false
            frameListener?.let { listener ->
                cameraStreamManager.removeFrameListener(listener)
            }
            Log.d(TAG, "Frame streaming to server stopped")
        }
    }

    private fun setupFrameListener() {
        try {
            frameListener = object : CameraFrameListener {
                override fun onFrame(
                    frameData: ByteArray,
                    offset: Int,
                    length: Int,
                    width: Int,
                    height: Int,
                    format: FrameFormat
                ) {
                    Log.d(TAG, "Frame received: width=$width, height=$height, length=$length")
                    if (isFrameStreaming && frameCounter++ % FRAME_SKIP == 0) {
                        Log.d(TAG, "Processing frame ${frameCounter/FRAME_SKIP}")
                        // Create a copy of the frame data
                        val data = ByteArray(length)
                        System.arraycopy(frameData, offset, data, 0, length)

                        // Process and send the frame
                        sendFrameToServer(data, width, height, format)
                    } else {
                        Log.d(TAG, "Skipping frame $frameCounter")
                    }
                }
            }

            // Request YUV420_888 format specifically
            cameraStreamManager.addFrameListener(
                ComponentIndexType.LEFT_OR_MAIN,
                FrameFormat.YUV420_888,
                frameListener!!
            )
            Log.d(TAG, "Frame listener setup successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up frame listener: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun sendFrameToServer(frameData: ByteArray, width: Int, height: Int, format: FrameFormat) {
        try {
            Log.d(TAG, "Starting to process frame for server")
            // Compress the frame
            val compressedFrame = ImageUtils.compressFrame(frameData, width, height, COMPRESSION_QUALITY)
            Log.d(TAG, "Frame compressed, size: ${compressedFrame.length}")

            val imuData = fragment?.getLatestImuData() ?: ImuData()
            Log.d(TAG, "Got IMU data")

            // Create JSON object with frame and IMU data
            val frameJson = JSONObject().apply {
                put("type", "video_frame")
                put("width", width)
                put("height", height)
                put("format", "JPEG")
                put("frame_index", frameIndex)
                put("data", compressedFrame)
                put("timestamp", System.currentTimeMillis())
                put("imu_data", JSONObject().apply {
                    put("velocity", JSONObject().apply {
                        put("x", imuData.velocity.x)
                        put("y", imuData.velocity.y)
                        put("z", imuData.velocity.z)
                    })
                    put("attitude", JSONObject().apply {
                        put("pitch", imuData.attitude.pitch)
                        put("roll", imuData.attitude.roll)
                        put("yaw", imuData.attitude.yaw)
                    })
                    put("location", JSONObject().apply {
                        put("latitude", imuData.location.latitude)
                        put("longitude", imuData.location.longitude)
                    })
                    put("timestamp", imuData.timestamp)
                    put("gimbal_attitude", JSONObject().apply {
                        put("pitch", imuData.gimbalAttitude.pitch)
                        put("roll", imuData.gimbalAttitude.roll)
                        put("yaw", imuData.gimbalAttitude.yaw)
                        put("yaw_relative_to_aircraft", imuData.gimbalAttitude.yawRelativeToAircraft)
                    })
                    put("imu_calibration", JSONObject().apply {
                        put("state", imuData.imuCalibrationStatus.state)
                        put("required_orientations", JSONArray(imuData.imuCalibrationStatus.orientationsNeeded))
                        put("message", imuData.imuCalibrationStatus.message)
                    })
                    put("gimbal_calibration", JSONObject().apply {
                        put("state", imuData.gimbalCalibrationStatus.state)
                        put("progress", imuData.gimbalCalibrationStatus.progress)
                        put("message", imuData.gimbalCalibrationStatus.message)
                    })
                })
            }

            // Send to WebSocket
            WebsocketContainer.send(frameJson.toString())

            // Save locally if recording
            if (isRecording) {
                Log.d(TAG, "Recording is active, saving frame")
                if (videoDirectory == null) {
                    Log.e(TAG, "Video directory is null!")
                    return
                }

                try {
                    // Save frame as JPEG
                    val frameFile = java.io.File(videoDirectory, "frame_${frameIndex}.png")
                    val frameBytes = Base64.decode(compressedFrame, Base64.NO_WRAP)
                    Log.d(TAG, "Saving frame $frameIndex, size: ${frameBytes.size} bytes to ${frameFile.absolutePath}")
                    frameFile.writeBytes(frameBytes)

                    // Write IMU data (without the frame data to save space)
                    frameJson.remove("data")
                    val imuJsonString = if (frameIndex > 0) ",\n${frameJson.toString(2)}" else frameJson.toString(2)
                    Log.d(TAG, "Writing IMU data for frame $frameIndex")
                    recordingFile?.write(imuJsonString)
                    recordingFile?.flush()
                    
                    frameIndex++
                    Log.d(TAG, "Successfully saved frame and IMU data")
                } catch (e: Exception) {
                    Log.e(TAG, "Error saving frame/IMU data: ${e.message}")
                    e.printStackTrace()
                }
            } else {
                Log.d(TAG, "Recording is not active, skipping frame save")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in sendFrameToServer: ${e.message}")
            e.printStackTrace()
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopFrameStreaming()
    }

    fun setFragment(fragment: FirstFragment) {
        this.fragment = fragment
    }

    private fun getLogFile(context: android.content.Context): java.io.File {
        val logDir = java.io.File(context.getExternalFilesDir(null), "logs")
        logDir.mkdirs()
        return java.io.File(logDir, "app.log")
    }

    private fun log(context: android.content.Context, message: String, isError: Boolean = false) {
        if (isError) {
            Log.e(TAG, message)
        } else {
            Log.d(TAG, message)
        }
        logToFile(context, message)
    }

    private fun logToFile(context: android.content.Context, message: String) {
        try {
            val timestamp = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(java.util.Date())
            getLogFile(context).appendText("$timestamp: $message\n")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write to log file: ${e.message}")
        }
    }

    fun startRecording(context: android.content.Context) {
        Log.d(TAG, "Starting recording")
        // log isRecording
        Log.d(TAG, "isRecording: $isRecording")
        if (isRecording) return
        
        try {
            Log.d(TAG, "Creating directory for recording session")
            // Create directory for this recording session
            val timestamp = java.text.SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", java.util.Locale.US)
                .format(java.util.Date())
            val baseDir = java.io.File(context.getExternalFilesDir(null), "drone_recordings")
            val sessionDir = java.io.File(baseDir, timestamp)
            if (sessionDir.mkdirs()) {
                log(context, "Created directory: ${sessionDir.absolutePath}")
            } else {
                log(context, "Failed to create directory: ${sessionDir.absolutePath}", true)
            }
            
            Log.d(TAG, "Creating IMU data log file")
            // Create IMU data log file
            val imuLogFile = java.io.File(sessionDir, "imu_data.json")
            recordingFile = FileWriter(imuLogFile)
            recordingFile?.write("[\n") // Start JSON array
            
            Log.d(TAG, "Setting up video directory")
            // Set up video directory
            videoDirectory = java.io.File(sessionDir, "frames")
            videoDirectory?.mkdirs()
            
            Log.d(TAG, "Setting isRecording to true")
            isRecording = true
            frameIndex = 0
            
            Log.d(TAG, "Sending WebSocket notification")
            // Send WebSocket notification
            val recordingJson = JSONObject().apply {
                put("type", "recording_status")
                put("status", "started")
                put("timestamp", System.currentTimeMillis())
                put("recording_path", sessionDir.absolutePath)
            }
            WebsocketContainer.send(recordingJson.toString())
            
            log(context, "Started recording to ${sessionDir.absolutePath}")
        } catch (e: Exception) {
            log(context, "Error starting recording: ${e.message}", true)
            e.printStackTrace()
            stopRecording(context)
        }
    }

    fun stopRecording(context: android.content.Context) {
        Log.d(TAG, "Stopping recording")
        Log.d(TAG, "isRecording: $isRecording")
        if (!isRecording) return
        
        try {
            Log.d(TAG, "Closing IMU data log file")
            recordingFile?.write("\n]") // Close JSON array
            recordingFile?.close()
            recordingFile = null
            Log.d(TAG, "Closing video file")
            currentVideoFile?.close()
            currentVideoFile = null
            Log.d(TAG, "Setting video directory to null")
            videoDirectory = null
            Log.d(TAG, "Setting isRecording to false")
            isRecording = false
            
            Log.d(TAG, "Sending WebSocket notification")
            // Send WebSocket notification
            val recordingJson = JSONObject().apply {
                put("type", "recording_status")
                put("status", "stopped")
                put("timestamp", System.currentTimeMillis())
                put("total_frames", frameIndex)
            }
            WebsocketContainer.send(recordingJson.toString())
            
            log(context, "Recording stopped")
        } catch (e: Exception) {
            log(context, "Error stopping recording: ${e.message}", true)
        }
    }

    fun isRecording() = isRecording
}
