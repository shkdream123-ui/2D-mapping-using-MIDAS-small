package com.example.depthrenderermidassmall.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class DepthClassifier(
    activity: Activity,
    modelPath: String,
    device: Classifier.Device,
    numThreads: Int
) : Classifier(activity, device, numThreads, modelPath, null) {

    override val preprocessNormalizeOp: TensorOperator = object : TensorOperator {
        override fun apply(tensor: TensorBuffer): TensorBuffer {
            val buffer = tensor.floatArray
            for (i in buffer.indices) buffer[i] = buffer[i] / 255.0f
            tensor.loadArray(buffer)
            return tensor
        }
    }
    override val postprocessNormalizeOp: TensorOperator? = null  // 필요 없으면 null

    fun runDepth(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        // 모델 입력용 Bitmap 변환
        val input = preprocess(bitmap) // loadImage 대신 직접 전처리 함수 사용 가능

        // 모델 추론
        tflite!!.run(input.buffer, outputProbabilityBuffer.buffer.rewind())

        val outputArray = outputProbabilityBuffer.floatArray
        val width = outputProbabilityBuffer.shape[2]
        val height = outputProbabilityBuffer.shape[1]

        val minVal = outputArray.minOrNull() ?: 0f
        val maxVal = outputArray.maxOrNull() ?: 1f
        val range = (maxVal - minVal).takeIf { it > 0 } ?: 1f

        Log.d(TAG, "Depth output min=$minVal, max=$maxVal")

        val depthBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val raw = outputArray[y * width + x]
                val norm = ((raw - minVal) / range * 255f).toInt().coerceIn(0, 255)
                val color = 0xFF shl 24 or (norm shl 16) or (norm shl 8) or norm
                depthBitmap.setPixel(x, y, color)
            }
        }

        // 화면 해상도에 맞춰 스케일링
        return Bitmap.createScaledBitmap(depthBitmap, targetWidth, targetHeight, false)
    }

    fun runRawDepth(bitmap: Bitmap): FloatArray {
        val input = preprocess(bitmap)
        tflite!!.run(input.buffer, outputProbabilityBuffer.buffer.rewind())
        return outputProbabilityBuffer.floatArray.clone() // 복사본 리턴
    }

    private fun preprocess(bitmap: Bitmap): TensorImage {
        // TensorFlow Lite 지원 클래스 사용
        val image = TensorImage(DataType.FLOAT32)
        val inputHeight = 256
        val inputWidth = 256
        image.load(bitmap)
        // 필요 시 모델 입력 크기에 맞게 리사이즈
        val resizeOp = ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR)
        return ImageProcessor.Builder()
            .add(resizeOp)
            .build()
            .process(image)
    }



    init {
        // --- 모델 입력 스펙 로그 ---
        val inputTensor = tflite!!.getInputTensor(0)
        val inputShape = inputTensor.shape()   // 예: [1, 256, 256, 3]
        val inputType = inputTensor.dataType() // 예: FLOAT32
        Log.d(TAG, "Depth model input shape: ${inputShape.joinToString(", ")}")
        Log.d(TAG, "Depth model input data type: $inputType")
    }

    companion object {
        private const val TAG = "DepthClassifier"
    }

    fun runDepthCombined(bitmap: Bitmap, outWidth: Int, outHeight: Int): Pair<FloatArray, Bitmap> {
        // 1) 전처리 및 추론
        val input = preprocess(bitmap) // TensorImage
        tflite!!.run(input.buffer, outputProbabilityBuffer.buffer.rewind())

        // 2) 출력 float 배열 얻기
        val rawOut = outputProbabilityBuffer.floatArray.clone() // 안전하게 clone

        val width = outputProbabilityBuffer.shape[2]
        val height = outputProbabilityBuffer.shape[1]

        // 3) 기존 depthClassifier 시각화 방식 적용
        val depthBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        // 기존 DepthClassifier 방식: 모델 출력 범위를 자동으로 감지하고 0..255로 스케일링
        val minVal = rawOut.minOrNull() ?: 0f
        val maxVal = rawOut.maxOrNull() ?: 1f
        val range = (maxVal - minVal).takeIf { it != 0f } ?: 1f

        for (y in 0 until height) {
            for (x in 0 until width) {
                val raw = rawOut[y * width + x]
                val normalized = ((raw - minVal) / range * 255f).toInt().coerceIn(0, 255)
                val color = (0xFF shl 24) or (normalized shl 16) or (normalized shl 8) or normalized
                depthBitmap.setPixel(x, y, color)
            }
        }

        // 4) 출력 사이즈 맞춰서 스케일링
        val scaled = Bitmap.createScaledBitmap(depthBitmap, outWidth, outHeight, false)
        return Pair(rawOut, scaled)
    }


}
