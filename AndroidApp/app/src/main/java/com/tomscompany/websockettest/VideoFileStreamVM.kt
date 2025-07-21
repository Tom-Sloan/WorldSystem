package com.tomscompany.websockettest

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

class VideoFileStreamVM : ViewModel() {
    companion object {
        private const val TAG = "VideoFileStreamVM"
        private const val TIMEOUT_US = 10000L // 10ms timeout
    }
    
    data class StreamStats(
        val bitrate: Float = 0f,
        val fps: Float = 0f,
        val frameCount: Int = 0
    )
    
    val streamStats = MutableLiveData(StreamStats())
    val isStreaming = MutableLiveData(false)
    val isPaused = MutableLiveData(false)
    
    private var streamingJob: Job? = null
    private var mediaExtractor: MediaExtractor? = null
    private var currentPosition: Long = 0
    private var frameCount = 0
    private var lastStatsUpdate = System.currentTimeMillis()
    private var bytesStreamed = 0L
    
    fun startVideoFileStreaming(
        context: Context,
        fileName: String,
        onVideoData: (ByteArray) -> Unit
    ) {
        if (isStreaming.value == true) {
            Log.w(TAG, "Already streaming")
            return
        }
        
        streamingJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                // First check if file exists in app's files directory
                val internalFile = File(context.filesDir, fileName)
                
                if (!internalFile.exists()) {
                    // Try to copy from assets
                    try {
                        context.assets.open(fileName).use { inputStream ->
                            FileOutputStream(internalFile).use { outputStream ->
                                inputStream.copyTo(outputStream)
                            }
                        }
                        Log.d(TAG, "Copied video file from assets to: ${internalFile.absolutePath}")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to copy video from assets: ${e.message}")
                        
                        // Try other locations if asset copy fails
                        val possibleLocations = listOf(
                            File(context.getExternalFilesDir(null), fileName),
                            File("/storage/emulated/0/Download", fileName),
                            File(context.cacheDir, fileName)
                        )
                        
                        var found = false
                        for (file in possibleLocations) {
                            if (file.exists() && file.canRead()) {
                                // Copy to internal storage for faster access next time
                                file.copyTo(internalFile, overwrite = true)
                                Log.d(TAG, "Copied video file from ${file.absolutePath} to internal storage")
                                found = true
                                break
                            }
                        }
                        
                        if (!found) {
                            Log.e(TAG, "Video file not found in any location: $fileName")
                            withContext(Dispatchers.Main) {
                                isStreaming.value = false
                            }
                            return@launch
                        }
                    }
                }
                
                streamVideoFile(internalFile, onVideoData)
            } catch (e: Exception) {
                Log.e(TAG, "Error streaming video file", e)
                withContext(Dispatchers.Main) {
                    isStreaming.value = false
                    isPaused.value = false
                }
            }
        }
    }
    
    private suspend fun streamVideoFile(
        file: File,
        onVideoData: (ByteArray) -> Unit
    ) {
        withContext(Dispatchers.Main) {
            isStreaming.value = true
            isPaused.value = false
            frameCount = 0
            bytesStreamed = 0L
        }
        
        mediaExtractor = MediaExtractor().apply {
            setDataSource(file.absolutePath)
        }
        
        // Find video track
        var videoTrackIndex = -1
        val trackCount = mediaExtractor!!.trackCount
        
        for (i in 0 until trackCount) {
            val format = mediaExtractor!!.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("video/")) {
                videoTrackIndex = i
                break
            }
        }
        
        if (videoTrackIndex == -1) {
            Log.e(TAG, "No video track found")
            return
        }
        
        mediaExtractor!!.selectTrack(videoTrackIndex)
        
        // Seek to saved position if resuming
        if (currentPosition > 0) {
            mediaExtractor!!.seekTo(currentPosition, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
        }
        
        val format = mediaExtractor!!.getTrackFormat(videoTrackIndex)
        val frameRate = format.getInteger(MediaFormat.KEY_FRAME_RATE)
        val frameDuration = 1000000L / frameRate // microseconds per frame
        
        val buffer = ByteBuffer.allocate(1024 * 1024) // 1MB buffer
        var lastFrameTime = System.nanoTime() / 1000 // Convert to microseconds
        
        while (isStreaming.value == true) {
            // Check if paused
            if (isPaused.value == true) {
                delay(100) // Check every 100ms
                continue
            }
            
            val sampleSize = mediaExtractor!!.readSampleData(buffer, 0)
            
            if (sampleSize < 0) {
                // End of stream, loop back to beginning
                mediaExtractor!!.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                currentPosition = 0
                continue
            }
            
            // Get sample data
            val data = ByteArray(sampleSize)
            buffer.get(data)
            buffer.clear()
            
            // Save current position
            currentPosition = mediaExtractor!!.sampleTime
            
            // Send video data
            onVideoData(data)
            
            // Update stats
            frameCount++
            bytesStreamed += sampleSize
            updateStats()
            
            // Advance to next sample
            mediaExtractor!!.advance()
            
            // Frame timing to maintain proper playback speed
            val currentTime = System.nanoTime() / 1000
            val elapsedTime = currentTime - lastFrameTime
            
            if (elapsedTime < frameDuration) {
                delay((frameDuration - elapsedTime) / 1000) // Convert to milliseconds
            }
            
            lastFrameTime = System.nanoTime() / 1000
        }
        
        mediaExtractor?.release()
        mediaExtractor = null
        
        withContext(Dispatchers.Main) {
            isStreaming.value = false
            isPaused.value = false
        }
    }
    
    fun pauseStreaming() {
        isPaused.value = true
    }
    
    fun resumeStreaming() {
        isPaused.value = false
    }
    
    fun stopStreaming() {
        streamingJob?.cancel()
        mediaExtractor?.release()
        mediaExtractor = null
        isStreaming.value = false
        isPaused.value = false
        currentPosition = 0
    }
    
    private fun updateStats() {
        val now = System.currentTimeMillis()
        val elapsed = (now - lastStatsUpdate) / 1000f
        
        if (elapsed >= 1.0f) {
            val fps = frameCount / elapsed
            val bitrate = (bytesStreamed * 8) / elapsed
            
            Handler(Looper.getMainLooper()).post {
                streamStats.value = StreamStats(
                    bitrate = bitrate,
                    fps = fps,
                    frameCount = frameCount
                )
            }
            
            frameCount = 0
            bytesStreamed = 0L
            lastStatsUpdate = now
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        stopStreaming()
    }
}