package com.tomscompany.websockettest

import android.util.Log
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.util.concurrent.TimeUnit
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject
import android.os.Handler
import android.os.Looper

object VideoWebsocketContainer {
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .writeTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(5, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .connectTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    
    private const val TAG = "VideoWebSocket"
    private val connectionStatusListeners = mutableListOf<(Boolean) -> Unit>()
    private var reconnectAttempts = 0
    private const val MAX_RECONNECT_ATTEMPTS = 10
    private const val RECONNECT_DELAY = 5000L
    private val isConnected = AtomicBoolean(false)
    
    fun connect() {
        val (serverIp, serverPort) = WebsocketContainer.getServerAddress()
        val portSegment = if (serverPort.isNotBlank() && serverPort != "80" && serverPort != "443") ":$serverPort" else ""
        val url = "ws://$serverIp$portSegment/ws/video"
        
        val request = Request.Builder()
            .url(url)
            .build()
            
        webSocket = client.newWebSocket(request, createWebSocketListener())
        Log.d(TAG, "Attempting to connect to video endpoint: $url")
    }
    
    fun sendBinary(data: ByteArray) {
        if (isConnected.get()) {
            val success = webSocket?.send(data.toByteString(0, data.size)) ?: false
            if (success) {
                Log.d(TAG, "Video data sent successfully (size: ${data.size})")
            } else {
                Log.e(TAG, "Failed to send video data")
            }
        } else {
            Log.w(TAG, "Cannot send video data - not connected")
        }
    }
    
    fun sendVideoConfig(config: JSONObject) {
        if (isConnected.get()) {
            val message = JSONObject().apply {
                put("type", "video_config")
                put("config", config)
                put("timestamp", System.currentTimeMillis())
            }.toString()
            
            val success = webSocket?.send(message) ?: false
            if (success) {
                Log.d(TAG, "Video config sent: $message")
            } else {
                Log.e(TAG, "Failed to send video config")
            }
        }
    }
    
    fun isConnected(): Boolean = isConnected.get()
    
    fun close() {
        webSocket?.close(1000, "Closing video WebSocket")
        webSocket = null
        isConnected.set(false)
        Log.d(TAG, "Video WebSocket closed")
    }
    
    fun addConnectionStatusListener(listener: (Boolean) -> Unit) {
        connectionStatusListeners.add(listener)
    }
    
    fun removeConnectionStatusListener(listener: (Boolean) -> Unit) {
        connectionStatusListeners.remove(listener)
    }
    
    private fun notifyConnectionStatusChanged(connected: Boolean) {
        isConnected.set(connected)
        connectionStatusListeners.forEach { it(connected) }
    }
    
    private fun scheduleReconnect() {
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            Handler(Looper.getMainLooper()).postDelayed({
                reconnectAttempts++
                Log.d(TAG, "Reconnecting video stream... (attempt $reconnectAttempts)")
                connect()
            }, RECONNECT_DELAY)
        } else {
            Log.e(TAG, "Max video reconnect attempts reached.")
        }
    }
    
    private fun createWebSocketListener() = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.d(TAG, "Video connection opened successfully")
            reconnectAttempts = 0
            notifyConnectionStatusChanged(true)
            
            // Send initial video configuration
            sendVideoConfig(JSONObject().apply {
                put("source", "dji_drone")
                put("format", "h264")
                put("resolution", "1920x1080")
            })
        }
        
        override fun onMessage(webSocket: WebSocket, text: String) {
            Log.d(TAG, "Received video control message: $text")
            // Handle any control messages from server if needed
        }
        
        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            Log.d(TAG, "Received binary message on video socket: ${bytes.size} bytes")
            // We don't expect binary messages from server on video socket
        }
        
        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            Log.d(TAG, "Video connection closing: $code / $reason")
            webSocket.close(1000, null)
            this@VideoWebsocketContainer.webSocket = null
            notifyConnectionStatusChanged(false)
            scheduleReconnect()
        }
        
        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.e(TAG, "Video WebSocket error: ${t.message}")
            t.printStackTrace()
            this@VideoWebsocketContainer.webSocket = null
            notifyConnectionStatusChanged(false)
            scheduleReconnect()
        }
    }
}