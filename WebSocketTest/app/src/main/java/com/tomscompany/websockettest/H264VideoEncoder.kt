package com.tomscompany.websockettest

import android.media.*
import android.os.Build
import android.os.Bundle
import android.util.Log
import java.nio.ByteBuffer

class H264VideoEncoder(
    private val width: Int,
    private val height: Int,
    private val bitrate: Int,
    private val frameRate: Int,
    private val iFrameInterval: Int = 1
) {
    companion object {
        private const val TAG = "H264VideoEncoder"
        private const val MIME_TYPE = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val TIMEOUT_US = 10000L
        
        // NAL unit types
        const val NAL_TYPE_NON_IDR = 1
        const val NAL_TYPE_IDR = 5
        const val NAL_TYPE_SEI = 6
        const val NAL_TYPE_SPS = 7
        const val NAL_TYPE_PPS = 8
        const val NAL_TYPE_AUD = 9
    }
    
    private var encoder: MediaCodec? = null
    @Volatile
    private var isRunning = false
    private var frameCount = 0L
    private val encoderLock = Any()
    
    // Callbacks
    var onEncodedFrame: ((ByteArray, MediaCodec.BufferInfo) -> Unit)? = null
    var onFormatChanged: ((MediaFormat) -> Unit)? = null
    var onError: ((Exception) -> Unit)? = null
    
    // Store SPS/PPS for later use
    private var sps: ByteArray? = null
    private var pps: ByteArray? = null
    
    fun configure() {
        try {
            val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameInterval)
            
            // Low latency settings
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                setInteger(MediaFormat.KEY_LOW_LATENCY, 1)
                setInteger(MediaFormat.KEY_PRIORITY, 0) // Realtime priority
            }
            
            // Configure encoder profile for compatibility
            setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileBaseline)
            setInteger(MediaFormat.KEY_LEVEL, MediaCodecInfo.CodecProfileLevel.AVCLevel31)
        }
        
            val newEncoder = MediaCodec.createEncoderByType(MIME_TYPE)
            
            // Use async callback mode for better performance
            newEncoder.setCallback(object : MediaCodec.Callback() {
            override fun onInputBufferAvailable(codec: MediaCodec, index: Int) {
                // Input buffers available for encoding
            }
            
            override fun onOutputBufferAvailable(
                codec: MediaCodec,
                index: Int,
                info: MediaCodec.BufferInfo
            ) {
                handleEncodedOutput(codec, index, info)
            }
            
            override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
                Log.e(TAG, "Encoder error: ${e.message}", e)
                onError?.invoke(e)
                isRunning = false
            }
            
            override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) {
                Log.d(TAG, "Output format changed: $format")
                extractParameterSets(format)
                onFormatChanged?.invoke(format)
            }
        })
        
            newEncoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder = newEncoder
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure encoder", e)
            onError?.invoke(e)
            throw e
        }
    }
    
    fun start() {
        synchronized(encoderLock) {
            encoder?.let {
                it.start()
                isRunning = true
                frameCount = 0
            } ?: throw IllegalStateException("Encoder not configured")
        }
    }
    
    fun stop() {
        synchronized(encoderLock) {
            isRunning = false
            try {
                encoder?.stop()
                encoder?.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping encoder", e)
            } finally {
                encoder = null
            }
        }
    }
    
    fun encodeFrame(yuv420Data: ByteArray, timestamp: Long) {
        if (!isRunning) return
        
        synchronized(encoderLock) {
            encoder?.let { enc ->
                // In async mode, we should wait for onInputBufferAvailable callback
                // For now, we'll use a simple approach with try-catch
                try {
                    val inputBufferIndex = enc.dequeueInputBuffer(0) // Non-blocking
                    if (inputBufferIndex >= 0) {
                        val inputBuffer = enc.getInputBuffer(inputBufferIndex)
                        inputBuffer?.let { buffer ->
                            buffer.clear()
                            buffer.put(yuv420Data)
                            
                            // Request keyframe periodically
                            if (frameCount > 0 && frameCount % (frameRate * iFrameInterval) == 0L) {
                                requestKeyFrame()
                            }
                            
                            enc.queueInputBuffer(
                                inputBufferIndex,
                                0,
                                yuv420Data.size,
                                timestamp,
                                0 // No flags on input
                            )
                            
                            frameCount++
                        }
                    }
                    Unit // Explicitly return Unit to avoid if expression issue
                } catch (e: Exception) {
                    Log.e(TAG, "Error encoding frame", e)
                    onError?.invoke(e)
                }
            }
        }
    }
    
    private fun handleEncodedOutput(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
        val outputBuffer = codec.getOutputBuffer(index)
        if (outputBuffer != null && info.size > 0) {
            // Extract NAL units
            val encodedData = ByteArray(info.size)
            outputBuffer.position(info.offset)
            outputBuffer.limit(info.offset + info.size)
            outputBuffer.get(encodedData)
            
            // Process based on flags
            when {
                (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0 -> {
                    // This contains SPS/PPS
                    Log.d(TAG, "Received codec config (SPS/PPS)")
                    parseSPSPPS(encodedData)
                }
                (info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0 -> {
                    Log.d(TAG, "Keyframe encoded")
                }
            }
            
            onEncodedFrame?.invoke(encodedData, info)
        }
        
        codec.releaseOutputBuffer(index, false)
    }
    
    private fun extractParameterSets(format: MediaFormat) {
        // Extract SPS
        format.getByteBuffer("csd-0")?.let { spsBuffer ->
            sps = ByteArray(spsBuffer.remaining())
            spsBuffer.get(sps)
            Log.d(TAG, "SPS extracted: ${sps?.size} bytes")
        }
        
        // Extract PPS
        format.getByteBuffer("csd-1")?.let { ppsBuffer ->
            pps = ByteArray(ppsBuffer.remaining())
            ppsBuffer.get(pps)
            Log.d(TAG, "PPS extracted: ${pps?.size} bytes")
        }
    }
    
    private fun parseSPSPPS(data: ByteArray) {
        // Parse concatenated SPS/PPS from codec config
        var offset = 0
        while (offset < data.size - 4) {
            // Find start code (0x00 0x00 0x00 0x01)
            if (data[offset] == 0x00.toByte() && 
                data[offset + 1] == 0x00.toByte() && 
                data[offset + 2] == 0x00.toByte() && 
                data[offset + 3] == 0x01.toByte()) {
                
                val nalType = data[offset + 4].toInt() and 0x1F
                when (nalType) {
                    NAL_TYPE_SPS -> Log.d(TAG, "Found SPS at offset $offset")
                    NAL_TYPE_PPS -> Log.d(TAG, "Found PPS at offset $offset")
                }
            }
            offset++
        }
    }
    
    fun getParameterSets(): Pair<ByteArray?, ByteArray?> = Pair(sps, pps)
    
    fun requestKeyFrame() {
        synchronized(encoderLock) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                encoder?.let {
                    val params = Bundle()
                    params.putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0)
                    it.setParameters(params)
                }
            }
        }
    }
    
    fun updateBitrate(newBitrate: Int) {
        synchronized(encoderLock) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                encoder?.let {
                    val params = Bundle()
                    params.putInt(MediaCodec.PARAMETER_KEY_VIDEO_BITRATE, newBitrate)
                    it.setParameters(params)
                }
            }
        }
    }
    
    fun isConfigured(): Boolean = encoder != null
}