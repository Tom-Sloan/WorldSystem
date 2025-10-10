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
import org.json.JSONException

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
    private var serverPort = ""
    private var lastMessageHandler: ((String) -> Unit)? = null
    private var reconnectAttempts = 0
    private const val MAX_RECONNECT_ATTEMPTS = 10
    private const val RECONNECT_DELAY = 5000L
    private var messageHandler: WebsocketMessageHandler? = null

    fun getServerAddress(): Pair<String, String> = Pair(serverIp, serverPort)

    fun setServerAddress(ip: String, port: String) {
        serverIp = ip
        serverPort = port
    }

    private fun scheduleReconnect() {
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            Handler(Looper.getMainLooper()).postDelayed({
                reconnectAttempts++
                lastMessageHandler?.let { handler ->
                    Log.d(TAG, "Reconnecting... (attempt $reconnectAttempts)")
                    connect(handler)
                }
            }, RECONNECT_DELAY)
        } else {
            Log.e(TAG, "Max reconnect attempts reached.")
        }
    }

    fun connect(messageHandler: (String) -> Unit) {
        // Save the handler for potential reconnection
        lastMessageHandler = messageHandler

        val portSegment = if (serverPort.isNotBlank() && serverPort != "80" && serverPort != "443") ":$serverPort" else ""
        val url = "ws://$serverIp:5001/ws/phone"
        val request = Request.Builder()
            .url(url)
            .build()
        webSocket = client.newWebSocket(request, createWebSocketListener(messageHandler))
        Log.d(TAG, "Attempting to connect to $url")
    }


    fun sendBinary(data: ByteArray) {
        if (isConnected()) {
            val success = webSocket?.send(data.toByteString(0, data.size)) ?: false
            if (success) {
                Log.d(TAG, "Binary message sent successfully (size: ${data.size})")
            } else {
                Log.e(TAG, "Failed to send binary message")
            }
        }
    }


    fun sendLog(tag: String, message: String, level: String = "DEBUG") {
        if (isConnected()) {
            try {
                val logJson = JSONObject().apply {
                    put("type", "log")
                    put("level", level)
                    put("tag", tag)
                    put("message", message)
                    put("timestamp", System.currentTimeMillis())
                }.toString()
                
                // Send log directly without queuing for immediate feedback
                webSocket?.send(logJson)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send log via WebSocket: ${e.message}")
            }
        }
    }

    fun send(message: String) {
        val jsonMessage = if (!message.trim().startsWith("{")) {
            JSONObject().apply {
                put("type", "status_message")
                put("message", message)
                put("timestamp", SntpClient.getCurrentTimeNanos())
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
            put("timestamp", SntpClient.getCurrentTimeNanos())
        }.toString()
        send(jsonMessage)
    }

    fun sendError(message: String, error: String? = null) {
        val jsonMessage = JSONObject().apply {
            put("type", "error_message")
            put("message", message)
            put("error", error)
            put("timestamp", SntpClient.getCurrentTimeNanos())
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
            reconnectAttempts = 0  // Reset attempts on successful connection
            notifyConnectionStatusChanged(true)
            trySendMessages()
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            Log.d(TAG, "Received text message: $text")
            
            // Check if this is a command message that should be routed to the message handler
            try {
                val jsonObject = JSONObject(text)
                val messageType = jsonObject.optString("type", "")
                
                if (messageType in listOf("movement", "rotation", "camera", "flightmode")) {
                    // This is a command message, route it to the WebsocketMessageHandler
                    messageHandler?.let { this@WebsocketContainer.messageHandler?.processMessage(text) }
                    return
                }
            } catch (e: Exception) {
                // Not a valid JSON or not a command message
                Log.d(TAG, "Not a command message or JSON parsing failed: ${e.message}")
            }
            
            // For other messages, use the regular message handler
            messageHandler(text)
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            Log.d(TAG, "Received binary message: ${bytes.size} bytes")
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            Log.d(TAG, "Connection closing: $code / $reason")
            webSocket.close(1000, null)
            this@WebsocketContainer.webSocket = null
            notifyConnectionStatusChanged(false)
            // Schedule a reconnect if not an intentional close
            scheduleReconnect()
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.e(TAG, "WebSocket error: ${t.message}")
            t.printStackTrace()
            this@WebsocketContainer.webSocket = null
            notifyConnectionStatusChanged(false)
            // Schedule a reconnect after a delay
            scheduleReconnect()
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

    fun setMessageHandler(handler: WebsocketMessageHandler) {
        messageHandler = handler
    }
}
