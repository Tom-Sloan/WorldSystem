package com.tomscompany.websockettest

import android.util.Log
import okhttp3.*
import okio.ByteString
import java.util.concurrent.TimeUnit
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

object WebsocketContainer {
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)  // No timeout for reading responses, wait indefinitely
        .writeTimeout(0, TimeUnit.MILLISECONDS)  // No timeout for writing requests, wait indefinitely
        .pingInterval(5, TimeUnit.SECONDS)      // Frequent pings to keep connection alive
        .retryOnConnectionFailure(true)         // Automatically retry on connection failure
        .connectTimeout(0, TimeUnit.MILLISECONDS) // No timeout for initial connection
        .build()
    private val messageQueue = ConcurrentLinkedQueue<String>()
    private const val TAG = "WebSocket"
    private const val MAX_QUEUE_SIZE = 10 // Adjust this value based on your needs
    private val isSending = AtomicBoolean(false)
    private val connectionStatusListeners = mutableListOf<(Boolean) -> Unit>()
    private var serverIp = "134.117.167.139"
    private var serverPort = "5001"
    // private var serverIp = "192.168.1.144"

    fun getServerAddress(): Pair<String, String> = Pair(serverIp, serverPort)

    fun setServerAddress(ip: String, port: String) {
        serverIp = ip
        serverPort = port
    }

    fun connect(messageHandler: (String) -> Unit) {
        val url = "ws://$serverIp:$serverPort/ws/video"
        val request = Request.Builder()
            .url(url)
            .build()
        webSocket = client.newWebSocket(request, createWebSocketListener(messageHandler))
        Log.d(TAG, "Attempting to connect to $url")
    }

    fun send(message: String) {
        val jsonMessage = if (!message.trim().startsWith("{")) {
            JSONObject().apply {
                put("type", "status_message")
                put("message", message)
                put("timestamp", System.currentTimeMillis())
            }.toString()
        } else {
            message // Message is already JSON
        }

        if (messageQueue.size >= MAX_QUEUE_SIZE) {
            messageQueue.poll()
        }
        messageQueue.offer(jsonMessage)
        trySendMessages()
    }

    fun sendStatus(message: String) {
        val jsonMessage = JSONObject().apply {
            put("type", "status_message")
            put("message", message)
            put("timestamp", System.currentTimeMillis())
        }.toString()
        send(jsonMessage)
    }

    fun sendError(message: String, error: String? = null) {
        val jsonMessage = JSONObject().apply {
            put("type", "error_message")
            put("message", message)
            put("error", error)
            put("timestamp", System.currentTimeMillis())
        }.toString()
        send(jsonMessage)
    }

    private fun trySendMessages() {
        if (isSending.compareAndSet(false, true)) {
            while (isConnected() && messageQueue.isNotEmpty()) {
                val message = messageQueue.poll() ?: break
                val success = webSocket?.send(message) ?: false
                if (success) {
                    Log.d(TAG, "Message sent successfully (length: ${message.length})")
                } else {
                    Log.e(TAG, "Failed to send message")
                    messageQueue.offer(message) // Put the message back in the queue
                    break
                }
            }
            isSending.set(false)
        }
    }

    fun isConnected(): Boolean {
        return webSocket != null
    }

    fun close() {
        webSocket?.close(1000, "Closing WebSocket")
        webSocket = null
        Log.d(TAG, "WebSocket closed")
    }

    private fun createWebSocketListener(messageHandler: (String) -> Unit) = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.d(TAG, "Connection opened successfully")
            notifyConnectionStatusChanged(true)
            trySendMessages()
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            Log.d(TAG, "Received text message: $text")
            messageHandler(text)
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            Log.d(TAG, "Received binary message: ${bytes.size} bytes")
            // Handle binary messages if needed
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(1000, null)
            Log.d(TAG, "Connection closing: $code / $reason")
            notifyConnectionStatusChanged(false)
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.e(TAG, "WebSocket error: ${t.message}")
            t.printStackTrace()
            this@WebsocketContainer.webSocket = null
            notifyConnectionStatusChanged(false)
        }
    }

    fun addConnectionStatusListener(listener: (Boolean) -> Unit) {
        connectionStatusListeners.add(listener)
    }

    fun removeConnectionStatusListener(listener: (Boolean) -> Unit) {
        connectionStatusListeners.remove(listener)
    }

    private fun notifyConnectionStatusChanged(isConnected: Boolean) {
        connectionStatusListeners.forEach { it(isConnected) }
    }
}
