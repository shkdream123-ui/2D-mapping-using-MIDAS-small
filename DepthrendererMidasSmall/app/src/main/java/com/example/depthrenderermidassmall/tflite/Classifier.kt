package com.example.depthrenderermidassmall.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import org.tensorflow.lite.Interpreter
//import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.util.PriorityQueue
import kotlin.Comparator
import kotlin.collections.get
import kotlin.math.min

/** A classifier specialized to label images using TensorFlow Lite.  */
abstract class Classifier protected constructor(
    activity: Activity,
    device: Device,
    numThreads: Int,
    protected val modelPath: String,       // constructor로 받음
    protected val labelPath: String? = null
) {

    enum class Device {
        CPU,
        NNAPI,
        //GPU
    }

    var imageSizeX: Int
    var imageSizeY: Int

    //private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    protected var tflite: Interpreter?
    private val tfliteOptions = Interpreter.Options()
    protected var inputImageBuffer: TensorImage
    protected var outputProbabilityBuffer: TensorBuffer
    private var probabilityProcessor: TensorProcessor?


    init {
        val tfliteModel = FileUtil.loadMappedFile(activity, this.modelPath)
        when (device) {
            Device.NNAPI -> tfliteOptions.addDelegate(NnApiDelegate().also { nnApiDelegate = it })
            //Device.GPU -> tfliteOptions.addDelegate(GpuDelegate().also { gpuDelegate = it })
            Device.CPU -> {}
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel, tfliteOptions)

        val imageShape = tflite!!.getInputTensor(0).shape()
        if (imageShape[1] != imageShape[2]) {
            imageSizeY = imageShape[2]
            imageSizeX = imageShape[3]
        } else {
            imageSizeY = imageShape[1]
            imageSizeX = imageShape[2]
        }
        inputImageBuffer = TensorImage(tflite!!.getInputTensor(0).dataType())
        outputProbabilityBuffer =
            TensorBuffer.createFixedSize(
                tflite!!.getOutputTensor(0).shape(),
                tflite!!.getOutputTensor(0).dataType()
            )
        probabilityProcessor = postprocessNormalizeOp?.let { TensorProcessor.Builder().add(it).build() }

        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.")
    }

    fun recognizeImage(bitmap: Bitmap, sensorOrientation: Int): FloatArray? {
        Trace.beginSection("recognizeImage")
        Trace.beginSection("loadImage")
        val startTime = SystemClock.uptimeMillis()
        inputImageBuffer = loadImage(bitmap, sensorOrientation)
        Log.v(TAG, "Timecost to load the image: ${SystemClock.uptimeMillis() - startTime}")
        Trace.endSection()

        Trace.beginSection("runInference")
        val startInference = SystemClock.uptimeMillis()
        tflite!!.run(inputImageBuffer.buffer, outputProbabilityBuffer.buffer.rewind())
        Log.v(
            TAG,
            "Timecost to run model inference: ${SystemClock.uptimeMillis() - startInference}"
        )
        Trace.endSection()
        Trace.endSection()

        return outputProbabilityBuffer.floatArray
    }

    fun close() {
        tflite?.close(); tflite = null
        //gpuDelegate?.close(); gpuDelegate = null
        nnApiDelegate?.close(); nnApiDelegate = null
    }

    protected fun loadImage(bitmap: Bitmap, sensorOrientation: Int): TensorImage {
        inputImageBuffer.load(bitmap)
        val cropSize = min(bitmap.width, bitmap.height)
        val numRotation = sensorOrientation / 90
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(numRotation))
            .add(preprocessNormalizeOp)
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    //protected abstract val modelPath: String?
    //protected abstract val labelPath: String?
    protected abstract val preprocessNormalizeOp: TensorOperator?
    protected abstract val postprocessNormalizeOp: TensorOperator?



    companion object {
        const val TAG: String = "ClassifierWithSupport"


        fun createDepthModel(
            activity: Activity,
            modelPath: String,
            device: Device,
            numThreads: Int
        ): Classifier {
            return DepthClassifier(activity, modelPath, device, numThreads)
        }


    }

}