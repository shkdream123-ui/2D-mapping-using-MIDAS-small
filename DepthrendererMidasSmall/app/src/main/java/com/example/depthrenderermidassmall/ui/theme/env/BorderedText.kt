package com.example.depthrenderermidassmall.ui.theme.env

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Paint.Align
import android.graphics.Rect
import android.graphics.Typeface
import java.util.Vector


/** A class that encapsulates the tedious bits of rendering legible, bordered text onto a canvas.  */
class BorderedText(interiorColor: Int, exteriorColor: Int, textSize: Float) {
    private val interiorPaint: Paint
    private val exteriorPaint: Paint

    val textSize: Float

    /**
     * Creates a left-aligned bordered text object with a white interior, and a black exterior with
     * the specified text size.
     *
     * @param textSize text size in pixels
     */
    constructor(textSize: Float) : this(Color.WHITE, Color.BLACK, textSize)

    /**
     * Create a bordered text object with the specified interior and exterior colors, text size and
     * alignment.
     *
     * @param interiorColor the interior text color
     * @param exteriorColor the exterior text color
     * @param textSize text size in pixels
     */
    init {
        interiorPaint = Paint()
        interiorPaint.setTextSize(textSize)
        interiorPaint.setColor(interiorColor)
        interiorPaint.setStyle(Paint.Style.FILL)
        interiorPaint.setAntiAlias(false)
        interiorPaint.setAlpha(255)

        exteriorPaint = Paint()
        exteriorPaint.setTextSize(textSize)
        exteriorPaint.setColor(exteriorColor)
        exteriorPaint.setStyle(Paint.Style.FILL_AND_STROKE)
        exteriorPaint.setStrokeWidth(textSize / 8)
        exteriorPaint.setAntiAlias(false)
        exteriorPaint.setAlpha(255)

        this.textSize = textSize
    }

    fun setTypeface(typeface: Typeface?) {
        interiorPaint.setTypeface(typeface)
        exteriorPaint.setTypeface(typeface)
    }

    fun drawText(canvas: Canvas, posX: Float, posY: Float, text: String) {
        canvas.drawText(text, posX, posY, exteriorPaint)
        canvas.drawText(text, posX, posY, interiorPaint)
    }

    fun drawLines(canvas: Canvas, posX: Float, posY: Float, lines: Vector<String>) {
        var lineNum = 0
        for (line in lines) {
            drawText(canvas, posX, posY - this.textSize * (lines.size - lineNum - 1), line)
            ++lineNum
        }
    }

    fun setInteriorColor(color: Int) {
        interiorPaint.setColor(color)
    }

    fun setExteriorColor(color: Int) {
        exteriorPaint.setColor(color)
    }

    fun setAlpha(alpha: Int) {
        interiorPaint.setAlpha(alpha)
        exteriorPaint.setAlpha(alpha)
    }

    fun getTextBounds(
        line: String?, index: Int, count: Int, lineBounds: Rect?
    ) {
        interiorPaint.getTextBounds(line, index, count, lineBounds)
    }

    fun setTextAlign(align: Align?) {
        interiorPaint.setTextAlign(align)
        exteriorPaint.setTextAlign(align)
    }
}
