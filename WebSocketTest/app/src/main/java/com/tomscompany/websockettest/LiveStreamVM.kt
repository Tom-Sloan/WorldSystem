package com.tomscompany.websockettest

import androidx.lifecycle.ViewModel
import android.util.Log
import android.view.Surface
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.manager.interfaces.ICameraStreamManager
import org.json.JSONObject
import dji.v5.manager.interfaces.ICameraStreamManager.CameraFrameListener
import dji.v5.manager.interfaces.ICameraStreamManager.FrameFormat
import dji.v5.common.callback.CommonCallbacks
import dji.v5.manager.datacenter.MediaDataCenter
import com.tomscompany.websockettest.data.ImuData
import java.io.FileWriter
import org.json.JSONArray
import dji.v5.manager.datacenter.camera.StreamInfo
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.LiveData
import java.nio.ByteBuffer
import java.nio.ByteOrder


class LiveStreamVM : ViewModel() {
    private val TAG = "LiveStreamVM"
    private val cameraStreamManager: ICameraStreamManager = MediaDataCenter.getInstance().cameraStreamManager
    private var isFrameStreaming = false
    private var frameListener: CameraFrameListener? = null
    private var frameCounter = 0
    private val FRAME_SKIP = 2 // Send every 2nd frame
    private val COMPRESSION_QUALITY = 100 // JPEG compression quality (0-100)
    private var fragment: FirstFragment? = null
    private var isRecording = false
    private var recordingFile: FileWriter? = null
    private var videoDirectory: java.io.File? = null
    private var currentVideoFile: java.io.FileOutputStream? = null
    private var frameIndex = 0
    private val _streamInfoLiveData = MutableLiveData<StreamInfo>()
    val streamInfoLiveData: LiveData<StreamInfo> get() = _streamInfoLiveData

    // Resolution settings
    enum class StreamResolution(val displayName: String, val widthPercent: Float, val heightPercent: Float) {
        ORIGINAL("Original (100%)", 1.0f, 1.0f),
        HALF("Half (50%)", 0.5f, 0.5f),
        QUARTER("Quarter (25%)", 0.25f, 0.25f),
        CUSTOM_720P("720p", -1.0f, -1.0f), // Custom resolution (1280x720)
        CUSTOM_480P("480p", -1.0f, -1.0f)  // Custom resolution (640x480)
    }

    private val _currentResolution = MutableLiveData<StreamResolution>(StreamResolution.ORIGINAL)
    val currentResolution: LiveData<StreamResolution> get() = _currentResolution

    // This property will be updated with real stream info on every frame.
    private var streamInfo: StreamInfo? = null
    // Used to calculate a frame rate (ms between frames)
    private var lastFrameTimestamp: Long = 0

    init {
        // You might add other initialization logic here if needed.
    }

    // Method to change resolution
    fun setResolution(resolution: StreamResolution) {
        if (_currentResolution.value != resolution) {
            _currentResolution.value = resolution
            Log.d(TAG, "Stream resolution changed to: ${resolution.displayName}")
            
            // Notify via WebSocket about resolution change
            val resolutionJson = JSONObject().apply {
                put("type", "resolution_change")
                put("resolution", resolution.displayName)
                put("timestamp", System.currentTimeMillis())
            }
            WebsocketContainer.send(resolutionJson.toString())
        }
    }

    /**
     * A custom implementation of the DJI StreamInfo.
     * In this example we subclass StreamInfo and override the getter methods.
     */
    class CustomStreamInfo(
        private val width: Int,
        private val height: Int,
        private val frameRate: Int,
        private val presentationTimeMs: Long,
        private val mimeType: ICameraStreamManager.MimeType,
        private val keyFrame: Boolean
    ) : StreamInfo() {
        override fun getWidth(): Int = width
        override fun getHeight(): Int = height
        override fun getFrameRate(): Int = frameRate
        override fun getPresentationTimeMs(): Long = presentationTimeMs
        override fun getMimeType(): ICameraStreamManager.MimeType = mimeType
        override fun isKeyFrame(): Boolean = keyFrame
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
                    
                    // Update streamInfo with details from this frame.
                    val currentTime = System.currentTimeMillis()
                    val computedFrameRate = if (lastFrameTimestamp != 0L) {
                        (1000 / (currentTime - lastFrameTimestamp)).toInt()
                    } else {
                        0
                    }
                    lastFrameTimestamp = currentTime
                    val newInfo = CustomStreamInfo(
                        width = width,
                        height = height,
                        frameRate = computedFrameRate,
                        presentationTimeMs = currentTime,
                        mimeType = ICameraStreamManager.MimeType.H264,
                        keyFrame = true
                    )
                    streamInfo = newInfo
                    _streamInfoLiveData.postValue(newInfo)


                    if (isFrameStreaming && frameCounter++ % FRAME_SKIP == 0) {
                        Log.d(TAG, "Processing frame ${frameCounter / FRAME_SKIP}")
                        // Create a copy of the frame data
                        val data = ByteArray(length)
                        System.arraycopy(frameData, offset, data, 0, length)

                        // Process and send the frame
                        sendFrameToServer(data, width, height)
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

    private fun sendFrameToServer(frameData: ByteArray, width: Int, height: Int) {
        try {
            Log.d(TAG, "Starting to process frame for server")
            
            // Calculate target dimensions based on selected resolution
            val resolution = _currentResolution.value ?: StreamResolution.ORIGINAL
            val targetWidth: Int
            val targetHeight: Int
            
            when (resolution) {
                StreamResolution.ORIGINAL -> {
                    targetWidth = width
                    targetHeight = height
                }
                StreamResolution.HALF -> {
                    targetWidth = (width * 0.5f).toInt()
                    targetHeight = (height * 0.5f).toInt()
                }
                StreamResolution.QUARTER -> {
                    targetWidth = (width * 0.25f).toInt()
                    targetHeight = (height * 0.25f).toInt()
                }
                StreamResolution.CUSTOM_720P -> {
                    targetWidth = 1280
                    targetHeight = 720
                }
                StreamResolution.CUSTOM_480P -> {
                    targetWidth = 640
                    targetHeight = 480
                }
            }
            
            // Compress and resize the frame
            val compressedFrame: ByteArray = if (targetWidth == width && targetHeight == height) {
                // No resizing needed
                ImageUtils.compressFrame(frameData, width, height, COMPRESSION_QUALITY)
            } else {
                // Resize to target dimensions
                ImageUtils.compressAndResizeFrame(frameData, width, height, targetWidth, targetHeight, COMPRESSION_QUALITY)
            }
            
            Log.d(TAG, "Frame compressed and resized, original: ${width}x${height}, target: ${targetWidth}x${targetHeight}, size: ${compressedFrame.size} bytes")

            // Get the synchronized timestamp using SntpClient with nanosecond precision.
            val timestamp = SntpClient.getCurrentTimeNanos()

            // Convert the timestamp (Long) to an 8-byte array (big-endian order).
            val timestampBytes = ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN).putLong(timestamp).array()

            // Create a new byte array that holds the timestamp and then the compressed frame data.
            val dataWithTimestamp = ByteArray(timestampBytes.size + compressedFrame.size)
            System.arraycopy(timestampBytes, 0, dataWithTimestamp, 0, timestampBytes.size)
            System.arraycopy(compressedFrame, 0, dataWithTimestamp, timestampBytes.size, compressedFrame.size)

            // Send the combined data via WebSocket.
            WebsocketContainer.sendBinary(dataWithTimestamp)

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
        Log.d(TAG, "isRecording: $isRecording")
        if (isRecording) return

        try {
            Log.d(TAG, "Creating directory for recording session")
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
            val imuLogFile = java.io.File(sessionDir, "imu_data.json")
            recordingFile = FileWriter(imuLogFile)
            recordingFile?.write("[\n")

            Log.d(TAG, "Setting up video directory")
            videoDirectory = java.io.File(sessionDir, "frames")
            videoDirectory?.mkdirs()

            Log.d(TAG, "Setting isRecording to true")
            isRecording = true
            frameIndex = 0

            Log.d(TAG, "Sending WebSocket notification")
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
            recordingFile?.write("\n]")
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

    /**
     * Returns the current stream information.
     * If no stream info has been updated yet, returns a default StreamInfo instance.
     */
    fun getCameraStreamInfo(): StreamInfo {
        return streamInfo ?: StreamInfo()
    }
}
