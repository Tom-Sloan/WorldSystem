package com.tomscompany.websockettest

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Base64
import java.io.ByteArrayOutputStream

object ImageUtils {
    fun compressFrame(frame: ByteArray, width: Int, height: Int, quality: Int = 70): String {
        // Quick conversion and compression
        val nv21Data = yuv420ToNv21(frame, width, height)
        val yuvImage = YuvImage(nv21Data, ImageFormat.NV21, width, height, null)
        // Direct JPEG compression
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, out)
        val jpegBytes = out.toByteArray()
        return Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
    }

    fun yuv420ToNv21(yuv420sp: ByteArray, width: Int, height: Int): ByteArray {
        val frameSize = width * height
        val nv21 = ByteArray(frameSize + frameSize / 2)
        
        // Copy Y plane as-is
        System.arraycopy(yuv420sp, 0, nv21, 0, frameSize)
        
        // Get U and V planes
        val uStart = frameSize
        val vStart = frameSize + (frameSize / 4)
        
        var yp = frameSize // UV starting position in NV21 array
        for (j in 0 until height / 2) {
            for (i in 0 until width / 2) {
                // Get U and V values
                val v = yuv420sp[vStart + j * (width / 2) + i]
                val u = yuv420sp[uStart + j * (width / 2) + i]
                
                // NV21 format requires V then U (VU ordering)
                nv21[yp++] = v
                nv21[yp++] = u
            }
        }
        
        return nv21
    }
}
