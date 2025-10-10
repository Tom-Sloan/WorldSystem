package com.tomscompany.websockettest

import androidx.lifecycle.ViewModel
import android.util.Log
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.LiveData
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import dji.sdk.keyvalue.value.common.ComponentIndexType

class H264VideoStreamVM : ViewModel() {
    companion object {
        private const val TAG = "H264VideoStreamVM"
        
        // Default video encoding parameters
        private const val DEFAULT_BITRATE = 4_000_000 // 4 Mbps
        private const val DEFAULT_FPS = 30
        private const val VIDEO_IFRAME_INTERVAL = 1 // Keyframe every 1 second
        
        // Packet types for WebSocket protocol
        const val PACKET_TYPE_SPS = 0x01.toByte()
        const val PACKET_TYPE_PPS = 0x02.toByte()
        const val PACKET_TYPE_KEYFRAME = 0x03.toByte()
        const val PACKET_TYPE_FRAME = 0x04.toByte()
        const val PACKET_TYPE_CONFIG = 0x05.toByte()
    }
    
    private val cameraStreamManager: ICameraStreamManager = MediaDataCenter.getInstance().cameraStreamManager
    private var h264Encoder: H264VideoEncoder? = null
    @Volatile
    private var isStreaming = false
    private var frameListener: ICameraStreamManager.CameraFrameListener? = null
    private var currentCallback: ((ByteArray) -> Unit)? = null
    private val streamingLock = Any()
    
    // Dynamic video parameters
    private var currentWidth = 0
    private var currentHeight = 0
    private var encoderInitialized = false
    
    // Buffer pool for memory efficiency
    private var bufferPool: BufferPool? = null
    
    // Statistics - use atomic for thread safety
    @Volatile private var totalBytesStreamed = 0L
    @Volatile private var streamStartTime = 0L
    @Volatile private var frameCount = 0L
    @Volatile private var keyFrameCount = 0L
    
    private val _streamStats = MutableLiveData<StreamStats>()
    val streamStats: LiveData<StreamStats> get() = _streamStats
    
    data class StreamStats(
        val bitrate: Double,
        val fps: Double,
        val keyFrames: Long,
        val totalFrames: Long,
        val totalBytes: Long
    )
    
    enum class StreamError {
        ENCODER_ERROR,
        NETWORK_ERROR,
        RESOLUTION_CHANGE,
        FRAME_TIMEOUT,
        NO_FRAMES_RECEIVED
    }
    
    private val _streamError = MutableLiveData<StreamError?>()
    val streamError: LiveData<StreamError?> get() = _streamError
    
    // Error handling
    private var consecutiveErrors = 0
    private val MAX_CONSECUTIVE_ERRORS = 5
    private var lastFrameTime = 0L
    private val FRAME_TIMEOUT_MS = 5000L
    
    fun startH264Streaming(onVideoData: (ByteArray) -> Unit) {
        synchronized(streamingLock) {
            if (isStreaming) {
                Log.w(TAG, "Already streaming")
                return
            }
            
            currentCallback = onVideoData
            encoderInitialized = false
            
            try {
                // Setup frame listener first to get actual resolution
                setupFrameListener()
                
                isStreaming = true
                streamStartTime = System.currentTimeMillis()
                
                // Reset statistics
                totalBytesStreamed = 0L
                frameCount = 0L
                keyFrameCount = 0L
                consecutiveErrors = 0
                lastFrameTime = System.currentTimeMillis()
                _streamError.postValue(null)
                
                // Start frame timeout monitor
                startFrameTimeoutMonitor()
                
                Log.d(TAG, "H.264 streaming started, waiting for first frame to initialize encoder")
                WebsocketContainer.sendLog(TAG, "H.264 streaming started, waiting for first frame", "INFO")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start H.264 streaming", e)
                stopH264Streaming()
                throw e
            }
        }
    }
    
    fun stopH264Streaming() {
        synchronized(streamingLock) {
            if (!isStreaming) return
            
            isStreaming = false
            
            frameListener?.let { listener ->
                try {
                    cameraStreamManager.removeFrameListener(listener)
                } catch (e: Exception) {
                    Log.e(TAG, "Error removing frame listener", e)
                }
            }
            frameListener = null
            
            h264Encoder?.stop()
            h264Encoder = null
            
            bufferPool?.clear()
            bufferPool = null
            currentCallback = null
            encoderInitialized = false
            
            Log.d(TAG, "H.264 streaming stopped")
            logFinalStats()
        }
    }
    
    private fun setupFrameListener() {
        Log.d(TAG, "Setting up frame listener for camera stream")
        WebsocketContainer.sendLog(TAG, "Setting up frame listener for camera stream", "INFO")
        frameListener = object : ICameraStreamManager.CameraFrameListener {
            override fun onFrame(
                frameData: ByteArray,
                offset: Int,
                length: Int,
                width: Int,
                height: Int,
                format: ICameraStreamManager.FrameFormat
            ) {
                if (!isStreaming) return
                
                Log.v(TAG, "Received frame: ${width}x${height}, format: $format, length: $length")
                
                // Send frame received log only for first frame or every 30th frame to avoid spam
                if (frameCount == 0L || frameCount % 30 == 0L) {
                    WebsocketContainer.sendLog(TAG, "Frame received: ${width}x${height}, format: $format, frameCount: $frameCount", "DEBUG")
                }
                
                try {
                    // Initialize encoder on first frame with actual resolution
                    if (!encoderInitialized || width != currentWidth || height != currentHeight) {
                        Log.d(TAG, "Initializing encoder for resolution: ${width}x${height}")
                        initializeEncoder(width, height)
                    }
                    
                    // Get buffers from pool
                    val pool = bufferPool ?: return
                    val yuv420Data = pool.acquire()
                    val nv12Data = pool.acquire()
                    
                    try {
                        System.arraycopy(frameData, offset, yuv420Data, 0, length)
                        
                        // Convert to NV12 format for MediaCodec
                        ImageUtils.convertYuv420ToNv12(yuv420Data, nv12Data, width, height)
                        
                        // Encode frame with proper timestamp
                        val timestamp = SystemClock.elapsedRealtimeNanos() / 1000 // Convert to microseconds
                        h264Encoder?.encodeFrame(nv12Data, timestamp)
                        
                        // Update last frame time and reset error count on successful processing
                        lastFrameTime = System.currentTimeMillis()
                        if (consecutiveErrors > 0) {
                            consecutiveErrors = 0
                            _streamError.postValue(null)
                        }
                    } finally {
                        // Always return buffers to pool
                        pool.release(yuv420Data)
                        pool.release(nv12Data)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing frame", e)
                }
            }
        }
        
        // Add listener for YUV420_888 format
        frameListener?.let { listener ->
            try {
                Log.d(TAG, "Adding frame listener to cameraStreamManager")
                WebsocketContainer.sendLog(TAG, "Adding frame listener to cameraStreamManager", "INFO")
                cameraStreamManager.addFrameListener(
                    ComponentIndexType.LEFT_OR_MAIN,
                    ICameraStreamManager.FrameFormat.YUV420_888,
                    listener
                )
                Log.d(TAG, "Frame listener added successfully")
                WebsocketContainer.sendLog(TAG, "Frame listener added successfully", "INFO")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to add frame listener", e)
                WebsocketContainer.sendLog(TAG, "Failed to add frame listener: ${e.message}", "ERROR")
                throw e
            }
        }
    }
    
    private fun handleEncodedFrame(
        data: ByteArray,
        info: MediaCodec.BufferInfo,
        onVideoData: (ByteArray) -> Unit
    ) {
        frameCount++
        
        val isKeyFrame = (info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0
        if (isKeyFrame) keyFrameCount++
        
        // Create packet with header
        val packet = createVideoPacket(
            data,
            info,
            if (isKeyFrame) PACKET_TYPE_KEYFRAME else PACKET_TYPE_FRAME
        )
        
        // Send packet
        onVideoData(packet)
        
        // Update statistics
        totalBytesStreamed += packet.size
        updateStreamStats()
    }
    
    private fun sendConfiguration(format: MediaFormat, onVideoData: (ByteArray) -> Unit) {
        h264Encoder?.getParameterSets()?.let { (sps, pps) ->
            sps?.let { 
                val spsPacket = createConfigPacket(it, PACKET_TYPE_SPS)
                onVideoData(spsPacket)
                Log.d(TAG, "Sent SPS: ${it.size} bytes")
            }
            
            pps?.let {
                val ppsPacket = createConfigPacket(it, PACKET_TYPE_PPS)
                onVideoData(ppsPacket)
                Log.d(TAG, "Sent PPS: ${it.size} bytes")
            }
        }
    }
    
    private fun createVideoPacket(data: ByteArray, info: MediaCodec.BufferInfo, packetType: Byte): ByteArray {
        // Packet structure:
        // [4 bytes: total packet size]
        // [1 byte: packet type]
        // [8 bytes: timestamp (microseconds)]
        // [4 bytes: data size]
        // [4 bytes: flags]
        // [N bytes: H.264 data]
        
        val headerSize = 21
        val totalSize = headerSize + data.size
        
        return ByteBuffer.allocate(totalSize).apply {
            order(ByteOrder.BIG_ENDIAN)
            putInt(totalSize)                    // Total packet size
            put(packetType)                      // Packet type
            putLong(info.presentationTimeUs)     // Timestamp
            putInt(data.size)                    // Data size
            putInt(info.flags)                   // MediaCodec flags
            put(data)                            // H.264 data
        }.array()
    }
    
    private fun createConfigPacket(data: ByteArray, packetType: Byte): ByteArray {
        val headerSize = 13
        val totalSize = headerSize + data.size
        
        return ByteBuffer.allocate(totalSize).apply {
            order(ByteOrder.BIG_ENDIAN)
            putInt(totalSize)                    // Total packet size
            put(packetType)                      // Packet type
            putLong(System.nanoTime() / 1000)    // Timestamp
            put(data)                            // SPS/PPS data
        }.array()
    }
    
    private fun updateStreamStats() {
        val elapsedMs = System.currentTimeMillis() - streamStartTime
        if (elapsedMs > 0) {
            val bitrate = (totalBytesStreamed * 8 * 1000) / elapsedMs.toDouble()
            val fps = (frameCount * 1000) / elapsedMs.toDouble()
            
            _streamStats.postValue(StreamStats(
                bitrate = bitrate,
                fps = fps,
                keyFrames = keyFrameCount,
                totalFrames = frameCount,
                totalBytes = totalBytesStreamed
            ))
        }
    }
    
    private fun logFinalStats() {
        val elapsedMs = System.currentTimeMillis() - streamStartTime
        val avgBitrate = if (elapsedMs > 0) (totalBytesStreamed * 8 * 1000) / elapsedMs else 0
        val avgFps = if (elapsedMs > 0) (frameCount * 1000) / elapsedMs else 0
        
        Log.d(TAG, """
            H.264 Stream Statistics:
            Duration: ${elapsedMs / 1000}s
            Total Frames: $frameCount
            Key Frames: $keyFrameCount
            Total Bytes: $totalBytesStreamed
            Average Bitrate: ${avgBitrate / 1_000_000} Mbps
            Average FPS: $avgFps
        """.trimIndent())
    }
    
    fun isStreaming() = isStreaming
    
    fun requestKeyFrame() {
        h264Encoder?.requestKeyFrame()
    }
    
    fun validateEncoderSupport(): Boolean {
        return try {
            // Test with common resolution
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, 1920, 1080)
            val codecList = MediaCodecList(MediaCodecList.REGULAR_CODECS)
            codecList.findEncoderForFormat(format) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error validating encoder support", e)
            false
        }
    }
    
    private fun initializeEncoder(width: Int, height: Int) {
        synchronized(streamingLock) {
            try {
                // Stop existing encoder if resolution changed
                if (encoderInitialized && (width != currentWidth || height != currentHeight)) {
                    h264Encoder?.stop()
                    h264Encoder = null
                    bufferPool?.clear()
                    bufferPool = null
                }
                
                currentWidth = width
                currentHeight = height
                
                // Initialize buffer pool for YUV data
                val frameSize = width * height * 3 / 2
                if (bufferPool == null) {
                    bufferPool = BufferPool(frameSize, 5)
                }
                
                // Initialize encoder with actual resolution
                if (h264Encoder == null) {
                    h264Encoder = H264VideoEncoder(
                        width,
                        height,
                        DEFAULT_BITRATE,
                        DEFAULT_FPS,
                        VIDEO_IFRAME_INTERVAL
                    ).apply {
                        onEncodedFrame = { data, info ->
                            currentCallback?.let { callback ->
                                handleEncodedFrame(data, info, callback)
                            }
                        }
                        
                        onFormatChanged = { format ->
                            currentCallback?.let { callback ->
                                sendConfiguration(format, callback)
                            }
                        }
                        
                        onError = { error ->
                            Log.e(TAG, "Encoder error: $error")
                            // Could trigger reconnection or fallback to JPEG
                        }
                        
                        configure()
                        start()
                    }
                }
                
                encoderInitialized = true
                Log.d(TAG, "Encoder initialized for ${width}x${height}")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize encoder", e)
                throw e
            }
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        stopH264Streaming()
    }
    
    private fun startFrameTimeoutMonitor() {
        // Monitor for frame timeouts
        Handler(Looper.getMainLooper()).postDelayed(object : Runnable {
            override fun run() {
                if (isStreaming) {
                    val timeSinceLastFrame = System.currentTimeMillis() - lastFrameTime
                    if (timeSinceLastFrame > FRAME_TIMEOUT_MS) {
                        Log.e(TAG, "Frame timeout detected: ${timeSinceLastFrame}ms")
                        
                        // Check if we never received any frames
                        if (frameCount == 0L) {
                            Log.e(TAG, "No frames have been received from drone camera")
                            WebsocketContainer.sendLog(TAG, "NO FRAMES RECEIVED from drone camera after ${timeSinceLastFrame}ms", "ERROR")
                            _streamError.postValue(StreamError.NO_FRAMES_RECEIVED)
                            handleStreamError(StreamError.NO_FRAMES_RECEIVED)
                        } else {
                            _streamError.postValue(StreamError.FRAME_TIMEOUT)
                            handleStreamError(StreamError.FRAME_TIMEOUT)
                        }
                    }
                    // Continue monitoring
                    Handler(Looper.getMainLooper()).postDelayed(this, 1000)
                }
            }
        }, FRAME_TIMEOUT_MS)
    }
    
    private fun handleStreamError(error: StreamError) {
        consecutiveErrors++
        Log.e(TAG, "Stream error: $error (count: $consecutiveErrors)")
        
        when {
            consecutiveErrors >= MAX_CONSECUTIVE_ERRORS -> {
                Log.e(TAG, "Max consecutive errors reached, stopping stream")
                stopH264Streaming()
            }
            error == StreamError.RESOLUTION_CHANGE -> {
                Log.w(TAG, "Resolution changed, reinitializing encoder")
                encoderInitialized = false
            }
            error == StreamError.FRAME_TIMEOUT || error == StreamError.ENCODER_ERROR -> {
                Log.w(TAG, "Requesting keyframe for recovery")
                requestKeyFrame()
            }
        }
    }
    
    fun clearError() {
        _streamError.postValue(null)
        consecutiveErrors = 0
    }
}