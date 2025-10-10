package com.tomscompany.websockettest

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import android.util.Log

/**
 * Thread-safe buffer pool for reusing byte arrays to reduce garbage collection
 * during video streaming.
 */
class BufferPool(private val bufferSize: Int, private val maxPoolSize: Int = 5) {
    companion object {
        private const val TAG = "BufferPool"
    }
    
    private val pool = ConcurrentLinkedQueue<ByteArray>()
    private val poolSize = AtomicInteger(0)
    private val totalAllocated = AtomicInteger(0)
    private val poolHits = AtomicInteger(0)
    private val poolMisses = AtomicInteger(0)
    
    /**
     * Acquire a buffer from the pool. If none available, creates a new one.
     */
    fun acquire(): ByteArray {
        val buffer = pool.poll()
        return if (buffer != null) {
            poolSize.decrementAndGet()
            poolHits.incrementAndGet()
            buffer
        } else {
            poolMisses.incrementAndGet()
            val total = totalAllocated.incrementAndGet()
            if (total <= maxPoolSize * 2) { // Log only first few allocations
                Log.d(TAG, "Allocating new buffer. Total allocated: $total")
            }
            ByteArray(bufferSize)
        }
    }
    
    /**
     * Release a buffer back to the pool for reuse.
     * Only accepts buffers of the correct size.
     */
    fun release(buffer: ByteArray) {
        if (buffer.size == bufferSize) {
            val currentSize = poolSize.get()
            if (currentSize < maxPoolSize) {
                // Only clear first few bytes for security, not entire buffer (too expensive)
                // Video data is not sensitive after encoding
                if (buffer.size >= 1024) {
                    // Clear first 1KB only
                    for (i in 0 until 1024) {
                        buffer[i] = 0
                    }
                }
                
                pool.offer(buffer)
                poolSize.incrementAndGet()
            }
        }
    }
    
    /**
     * Clear all buffers from the pool.
     */
    fun clear() {
        pool.clear()
        poolSize.set(0)
        val stats = getStats()
        Log.d(TAG, "Pool cleared. Stats - Hits: ${stats.poolHits}, Misses: ${stats.poolMisses}, Total Allocated: ${stats.totalAllocated}")
    }
    
    /**
     * Get current pool statistics for debugging.
     */
    fun getStats(): PoolStats {
        val hits = poolHits.get()
        val misses = poolMisses.get()
        val total = hits + misses
        
        return PoolStats(
            currentSize = poolSize.get(),
            maxSize = maxPoolSize,
            bufferSize = bufferSize,
            totalAllocated = totalAllocated.get(),
            poolHits = hits,
            poolMisses = misses,
            hitRate = if (total > 0) hits.toFloat() / total else 0f
        )
    }
    
    data class PoolStats(
        val currentSize: Int,
        val maxSize: Int,
        val bufferSize: Int,
        val totalAllocated: Int,
        val poolHits: Int,
        val poolMisses: Int,
        val hitRate: Float
    )
}