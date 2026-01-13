package com.example.depthrenderermidassmall.ui.theme.customview

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import java.util.LinkedList


/** A simple View providing a render callback to other classes.  */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
    private val callbacks: MutableList<DrawCallback> = LinkedList<DrawCallback>()

    fun addCallback(callback: DrawCallback?) {
        callbacks.add(callback!!)
    }

    @Synchronized
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)  // View 기본 그리기 동작 유지
        for (callback in callbacks) {
            callback.drawCallback(canvas)
        }
    }

    /** Interface defining the callback for client classes.  */
    interface DrawCallback {
        fun drawCallback(canvas: Canvas?)
    }
}