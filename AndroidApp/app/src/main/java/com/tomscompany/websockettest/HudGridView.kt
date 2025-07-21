package com.tomscompany.websockettest

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class HudGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    private val gridPaint = Paint().apply {
        color = context.resources.getColor(R.color.hud_grid, null)
        strokeWidth = 1f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    private val gridSize = 20f // 20dp grid
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val gridSizePx = gridSize * resources.displayMetrics.density
        
        // Draw vertical lines
        var x = 0f
        while (x <= width) {
            canvas.drawLine(x, 0f, x, height.toFloat(), gridPaint)
            x += gridSizePx
        }
        
        // Draw horizontal lines
        var y = 0f
        while (y <= height) {
            canvas.drawLine(0f, y, width.toFloat(), y, gridPaint)
            y += gridSizePx
        }
    }
}