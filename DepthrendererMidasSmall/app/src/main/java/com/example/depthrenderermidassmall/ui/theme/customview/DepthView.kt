package com.example.depthrenderermidassmall.ui.theme.customview

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/*class DepthView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var depthBitmap: Bitmap? = null
    private val paint = Paint()

    fun setDepthMap(map: FloatArray) {
        depthMap = map
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val map = depthMap ?: return
        val size = Math.min(width, height)
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)

        var max = Float.NEGATIVE_INFINITY
        var min = Float.POSITIVE_INFINITY
        for (v in map) {
            if (v > max) max = v
            if (v < min) min = v
        }
        val scale = if (max - min > 0f) 255f / (max - min) else 1f

        for (y in 0 until size) {
            for (x in 0 until size) {
                val idx = y * size + x
                val value = if (idx < map.size) ((map[idx] - min) * scale).toInt() else 0
                val color = Color.rgb(value, value, value)
                bmp.setPixel(x, y, color)
            }
        }

        canvas.drawBitmap(bmp, 0f, 0f, paint)
    }
}*/
