package com.tomscompany.websockettest

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Base64
import java.io.ByteArrayOutputStream

object ImageUtils {
    fun compressFrame(frame: ByteArray, width: Int, height: Int, quality: Int = 70): ByteArray {
        val nv21Data = yuv420ToNv21(frame, width, height)
        val yuvImage = YuvImage(nv21Data, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, out)
        return out.toByteArray()
    }

    fun compressAndResizeFrame(frame: ByteArray, width: Int, height: Int, targetWidth: Int, targetHeight: Int, quality: Int = 70): ByteArray {
        if (targetWidth == width && targetHeight == height) {
            return compressFrame(frame, width, height, quality)
        }
        
        val nv21Data = yuv420ToNv21(frame, width, height)
        
        val yuvImage = YuvImage(nv21Data, ImageFormat.NV21, width, height, null)
        val tempOut = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, tempOut)
        
        val bitmapOptions = android.graphics.BitmapFactory.Options()
        bitmapOptions.inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(tempOut.toByteArray(), 0, tempOut.size(), bitmapOptions)
        
        val resizedBitmap = android.graphics.Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        bitmap.recycle()
        
        val out = ByteArrayOutputStream()
        resizedBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, quality, out)
        resizedBitmap.recycle()
        
        return out.toByteArray()
    }

    fun yuv420ToNv21(yuv420sp: ByteArray, width: Int, height: Int): ByteArray {
        val frameSize = width * height
        val nv21 = ByteArray(frameSize + frameSize / 2)
        
        System.arraycopy(yuv420sp, 0, nv21, 0, frameSize)
        
        val uStart = frameSize
        val vStart = frameSize + (frameSize / 4)
        
        var yp = frameSize
        for (j in 0 until height / 2) {
            for (i in 0 until width / 2) {
                val v = yuv420sp[vStart + j * (width / 2) + i]
                val u = yuv420sp[uStart + j * (width / 2) + i]
                
                nv21[yp++] = v
                nv21[yp++] = u
            }
        }
        
        return nv21
    }
    
    /**
     * Convert YUV420 planar format to NV12 semi-planar format for MediaCodec.
     * NV12 has Y plane followed by interleaved UV plane.
     * 
     * Note: DJI's YUV420_888 format typically provides I420 (YUV420P) layout:
     * Y plane: [0, width*height)
     * U plane: [width*height, width*height + width*height/4)
     * V plane: [width*height + width*height/4, width*height + width*height/2)
     */
    fun convertYuv420ToNv12(yuv420: ByteArray, nv12: ByteArray, width: Int, height: Int) {
        val frameSize = width * height
        val chromaSize = frameSize / 4
        
        // Validate input/output array sizes
        require(yuv420.size >= frameSize + frameSize / 2) { 
            "Input YUV420 array too small: ${yuv420.size} < ${frameSize + frameSize / 2}" 
        }
        require(nv12.size >= frameSize + frameSize / 2) { 
            "Output NV12 array too small: ${nv12.size} < ${frameSize + frameSize / 2}" 
        }
        
        // Copy Y plane (same in both formats)
        System.arraycopy(yuv420, 0, nv12, 0, frameSize)
        
        // Convert UV planes from planar to interleaved
        val uvStart = frameSize
        val uStart = frameSize
        val vStart = frameSize + chromaSize
        
        var uvIndex = uvStart
        for (i in 0 until chromaSize) {
            nv12[uvIndex++] = yuv420[uStart + i]  // U
            nv12[uvIndex++] = yuv420[vStart + i]  // V
        }
    }
    
    /**
     * Convert NV12 semi-planar format to YUV420 planar format.
     * This is the reverse of convertYuv420ToNv12.
     * 
     * NV12: Y plane followed by interleaved UV
     * YUV420P: Y plane, U plane, V plane
     */
    fun convertNv12ToYuv420(nv12: ByteArray, yuv420: ByteArray, width: Int, height: Int) {
        val frameSize = width * height
        val chromaSize = frameSize / 4
        
        // Validate array sizes
        require(nv12.size >= frameSize + frameSize / 2) { 
            "Input NV12 array too small" 
        }
        require(yuv420.size >= frameSize + frameSize / 2) { 
            "Output YUV420 array too small" 
        }
        
        // Copy Y plane
        System.arraycopy(nv12, 0, yuv420, 0, frameSize)
        
        // De-interleave UV planes
        val uvStart = frameSize
        val uStart = frameSize
        val vStart = frameSize + chromaSize
        
        var uvIndex = uvStart
        for (i in 0 until chromaSize) {
            yuv420[uStart + i] = nv12[uvIndex++]  // U
            yuv420[vStart + i] = nv12[uvIndex++]  // V
        }
    }
}
