package com.example.depthrenderermidassmall.ui.theme

import android.content.Context
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.depthrenderermidassmall.R

class CameraConnectionFragment(
    private val activity: Context,
    private val layoutId: Int,
    private val desiredSize: Size
) : Fragment() {

    interface FrameListener {
        fun onFrameAvailable(bitmap: Bitmap)
    }

    private var frameListener: FrameListener? = null
    private lateinit var previewView: TextureView
    private lateinit var cameraDevice: CameraDevice
    private lateinit var captureSession: CameraCaptureSession
    private lateinit var backgroundHandler: Handler
    private lateinit var backgroundThread: HandlerThread

    fun setFrameListener(listener: FrameListener) {
        frameListener = listener
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(layoutId, container, false)
        previewView = view.findViewById(R.id.previewTextureView) // xml id 그대로
        previewView.surfaceTextureListener = surfaceTextureListener
        Log.d("CameraDebug", "onCreateView: TextureView 준비됨")
        return view
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
        Log.d("CameraDebug", "onResume: 실행됨")
        if (previewView.isAvailable) {
            Log.d("CameraDebug", "onResume: TextureView 사용 가능 → 카메라 오픈 시도")
            openCamera()
        } else {
            previewView.surfaceTextureListener = surfaceTextureListener
        }
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            Log.d("CameraDebug", "SurfaceTexture available: ${width}x$height → 카메라 오픈 시도")
            if (width == 0 || height == 0) return
            openCamera()
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            Log.d("CameraDebug", "onSurfaceTextureUpdated: 새 프레임 도착")
            val bitmap = previewView.bitmap
            bitmap?.let { frameListener?.onFrameAvailable(it) }
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread.looper)
        Log.d("CameraDebug", "Background thread 시작됨")
    }

    private fun stopBackgroundThread() {
        backgroundThread.quitSafely()
        backgroundThread.join()
        Log.d("CameraDebug", "Background thread 종료됨")
    }

    private fun openCamera() {
        try {
            val manager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = manager.cameraIdList[0]
            Log.d("CameraDebug", "openCamera: 카메라ID = $cameraId")
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    Log.d("CameraDebug", "CameraDevice 열림")
                    cameraDevice = camera
                    createCaptureSession()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    Log.d("CameraDebug", "CameraDevice 연결 해제됨")
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e("CameraDebug", "CameraDevice 에러 발생: $error")
                    camera.close()
                }
            }, backgroundHandler)
        } catch (e: SecurityException) {
            Log.e("CameraDebug", "카메라 권한 문제 발생: ${e.message}")
        } catch (e: Exception) {
            Log.e("CameraDebug", "openCamera 실패: ${e.message}")
        }
    }

    private fun createCaptureSession() {
        val surfaceTexture = previewView.surfaceTexture
        if (surfaceTexture == null) {
            Log.e("CameraDebug", "createCaptureSession 실패: surfaceTexture null")
            return
        }
        surfaceTexture.setDefaultBufferSize(desiredSize.width, desiredSize.height)
        val surface = Surface(surfaceTexture)
        cameraDevice.createCaptureSession(
            listOf(surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    Log.d("CameraDebug", "CaptureSession 구성 완료")
                    captureSession = session
                    val captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    captureRequest.addTarget(surface)
                    captureRequest.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                    captureSession.setRepeatingRequest(captureRequest.build(), null, backgroundHandler)
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e("CameraDebug", "CaptureSession 구성 실패")
                }
            },
            backgroundHandler
        )
    }

    private fun closeCamera() {
        if (::captureSession.isInitialized) {
            captureSession.close()
            Log.d("CameraDebug", "CaptureSession 닫힘")
        }
        if (::cameraDevice.isInitialized) {
            cameraDevice.close()
            Log.d("CameraDebug", "CameraDevice 닫힘")
        }
    }
}
