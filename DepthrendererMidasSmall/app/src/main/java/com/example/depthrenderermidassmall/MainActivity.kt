package com.example.depthrenderermidassmall

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.TextureView
import android.widget.ImageView
import android.widget.Button
import com.example.depthrenderermidassmall.ui.theme.CameraActivity
import com.example.depthrenderermidassmall.ui.theme.CameraConnectionFragment
import com.example.depthrenderermidassmall.tflite.Classifier
import com.example.depthrenderermidassmall.tflite.DepthClassifier

class MainActivity : CameraActivity(), CameraConnectionFragment.FrameListener {

    //private lateinit var previewView: TextureView
    private lateinit var depthView: ImageView
    private lateinit var depthClassifier: DepthClassifier

    override val layoutId: Int
        get() = R.layout.tfe_ic_activity_camera

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.tfe_ic_activity_camera)
        //findViewById<Button>(R.id.btn_send_test).setOnClickListener {
            //sendTestPattern(256, 256)
        //}
        //previewView = findViewById(R.id.previewTextureView)
        depthView = findViewById(R.id.depthView)


        // DepthClassifier 초기화
        depthClassifier = DepthClassifier(
            this,
            "depth_model.tflite",       // 모델 파일명 (assets 폴더에 있어야 함)
            Classifier.Device.CPU,      // CPU / NNAPI 선택 가능
            4                            // 쓰레드 수
        )

        // 카메라 프래그먼트 추가
        val cameraFragment = CameraConnectionFragment(
            this,
            R.layout.tfe_ic_camera_connection_fragment,
            Size(640, 480)
        )
        cameraFragment.setFrameListener(this)

        supportFragmentManager.beginTransaction()
            .replace(R.id.container, cameraFragment)
            .commit()
    }

    override fun onFrameAvailable(bitmap: Bitmap) {
        val (rawDepth, depthBitmap) = depthClassifier.runDepthCombined(bitmap, 256, 256)


        //val rawDepth = depthClassifier.runRawDepth(bitmap)
        //Log.d("DepthCheck", "min=${rawDepth.minOrNull()}, max=${rawDepth.maxOrNull()}")
        val width = depthBitmap.width
        val height = depthBitmap.height

        var minGray = 255
        var maxGray = 0

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = depthBitmap.getPixel(x, y)
                val gray = pixel and 0xFF // ARGB 모두 동일하면 B 채널 사용
                if (gray < minGray) minGray = gray
                if (gray > maxGray) maxGray = gray
            }
        }

        Log.d("DepthCheck", "depthBitmap gray min=$minGray, max=$maxGray")

        // 기존 로그
        Log.d("DepthCheck", "depthBitmap: width=${depthBitmap.width}, height=${depthBitmap.height}")
        //val inputWidth = 256
        //val inputHeight = 256

        //val depthBitmap = depthClassifier.runDepth(bitmap, inputWidth, inputHeight)

        //sendFrameWithDepthNoCompression(depthClassifier, bitmap)
        sendFrameWithDepthNoCompression(bitmap, rawDepth)
        /*sendFrameWithDepthAndPose(
            bitmap = bitmap,
            depthArray = rawDepth,
            yaw = yawRad
        )*/

        runOnUiThread {
            depthView.setImageBitmap(depthBitmap)
        }
    }


    // DepthClassifier 출력 float 배열을 Bitmap으로 변환
    /*private fun convertDepthToBitmap(depthArray: Array<Array<FloatArray>>): Bitmap {
        val height = depthArray.size
        val width = depthArray[0].size
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                // 0~1 범위 -> 0~255 그레이 스케일
                val value = (depthArray[y][x][0] * 255).toInt().coerceIn(0, 255)
                val color = 0xFF shl 24 or (value shl 16) or (value shl 8) or value
                bmp.setPixel(x, y, color)
            }
        }
        return bmp
    }*/

}
