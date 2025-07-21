package com.tomscompany.websockettest

import android.util.Log
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

object SntpClient {
    private const val TAG = "SntpClient"
    private const val NTP_PORT = 123
    private const val NTP_PACKET_SIZE = 48
    private const val NTP_MODE_CLIENT = 3
    private const val NTP_VERSION = 3
    // Offset in the NTP packet where the transmit timestamp starts.
    private const val NTP_TRANSMIT_TIME_OFFSET = 40

    // The computed offset (in milliseconds) between the local system clock and the NTP server.
    var ntpTimeOffset: Long = 0
    
    // Base nano time when the NTP synchronization was performed
    private var baseNanoTime: Long = 0
    // NTP time in nanoseconds at the moment of synchronization
    private var baseNtpTimeNanos: Long = 0

    /**
     * Request the current time from an NTP server and compute the offset.
     *
     * @param ntpHost The NTP server hostname (default "pool.ntp.org").
     * @param timeout Timeout in milliseconds.
     * @return True if successful, false otherwise.
     */
    fun requestTime(ntpHost: String = "pool.ntp.org", timeout: Int = 30000): Boolean {
        try {
            val buffer = ByteArray(NTP_PACKET_SIZE)
            // Set mode = 3 (client) and version = 3.
            buffer[0] = (NTP_MODE_CLIENT or (NTP_VERSION shl 3)).toByte()

            val requestTime = System.currentTimeMillis()
            val requestTicks = System.nanoTime()

            val address = InetAddress.getByName(ntpHost)
            val packet = DatagramPacket(buffer, buffer.size, address, NTP_PORT)
            DatagramSocket().use { socket ->
                socket.soTimeout = timeout
                socket.send(packet)
                socket.receive(packet)
            }

            val responseTicks = System.nanoTime()
            val responseTime = requestTime + (responseTicks - requestTicks) / 1_000_000L

            // Extract transmit time (the time at which the NTP server sent its reply).
            val transmitTime = readTimeStamp(buffer, NTP_TRANSMIT_TIME_OFFSET)

            // Compute the offset: (server time) minus (local estimated time at reception)
            ntpTimeOffset = transmitTime - responseTime
            
            // Save reference points for nanosecond precision calculations
            baseNanoTime = System.nanoTime()
            baseNtpTimeNanos = (transmitTime * 1_000_000L) // convert millis to nanos

            Log.d(TAG, "NTP time offset: $ntpTimeOffset ms")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error requesting time from NTP server", e)
            return false
        }
    }

    /**
     * Reads an NTP timestamp (8 bytes) from the buffer starting at the given offset and converts
     * it from the NTP epoch (1900) to Unix epoch (1970) in milliseconds.
     */
    private fun readTimeStamp(buffer: ByteArray, offset: Int): Long {
        val seconds = ((buffer[offset].toLong() and 0xFF) shl 24) or
                ((buffer[offset + 1].toLong() and 0xFF) shl 16) or
                ((buffer[offset + 2].toLong() and 0xFF) shl 8) or
                (buffer[offset + 3].toLong() and 0xFF)
        val fraction = ((buffer[offset + 4].toLong() and 0xFF) shl 24) or
                ((buffer[offset + 5].toLong() and 0xFF) shl 16) or
                ((buffer[offset + 6].toLong() and 0xFF) shl 8) or
                (buffer[offset + 7].toLong() and 0xFF)
        // Convert seconds from NTP epoch to Unix epoch.
        val ntpSeconds = seconds - 2208988800L
        return ntpSeconds * 1000L + (fraction * 1000L) / 0x100000000L
    }

    /**
     * Returns the current (corrected) time in milliseconds using the computed NTP offset.
     */
    fun getCurrentTimeMillis(): Long {
        return System.currentTimeMillis() + ntpTimeOffset
    }
    
    /**
     * Returns the current (corrected) time in nanoseconds using the computed NTP offset.
     * This provides nanosecond precision relative to the synchronized time.
     */
    fun getCurrentTimeNanos(): Long {
        // If we haven't synchronized with NTP yet, return a reasonable value
        if (baseNanoTime == 0L) {
            return System.nanoTime()
        }
        
        // Calculate elapsed nanos since base measurement
        val elapsedNanos = System.nanoTime() - baseNanoTime
        
        // Return the NTP base time plus elapsed nanos
        return baseNtpTimeNanos + elapsedNanos
    }
}
