package com.example.depthrenderermidassmall.ui.theme

import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.depthrenderermidassmall.tflite.DepthClassifier
import java.io.OutputStream
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.Deflater

abstract class CameraActivity : AppCompatActivity() {

    protected var backgroundHandler: Handler? = null
    private var backgroundThread: HandlerThread? = null

    private var socket: Socket? = null
    private var outputStream: OutputStream? = null
    private var senderThread: Thread? = null

    private lateinit var sensorManager: SensorManager
    private var gyroSensor: Sensor? = null
    private var accelSensor: Sensor? = null


    abstract val layoutId: Int
    abstract fun onFrameAvailable(bitmap: Bitmap)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(layoutId)
        startBackgroundThread()
        startSocketSender()
    }

    override fun onResume() {
        super.onResume()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        if (backgroundThread == null) {
            startBackgroundThread()
        }

        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        gyroSensor?.let {
            sensorManager.registerListener(
                sensorListener,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }

        accelSensor?.let {
            sensorManager.registerListener(
                sensorListener,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    override fun onPause() {
        sensorManager.unregisterListener(sensorListener)
        stopBackgroundThread()
        super.onPause()
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        backgroundThread?.join()
        backgroundThread = null
        backgroundHandler = null
    }

    private fun startSocketSender() {
        senderThread = Thread {
            try {
                socket = Socket("192.168.35.157", 5000)// 🔹 PC의 IP/포트로 변경
                outputStream = socket!!.getOutputStream()
                Log.d("TCP", "Socket connected")
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        senderThread?.start()
    }

    private fun stopSocketSender() {
        try {
            outputStream?.close()
            socket?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        senderThread?.interrupt()
        senderThread = null
    }

    protected fun sendFrameWithDepth(classifier: DepthClassifier, bitmap: Bitmap) {
        Log.d("TCP", "sendFrameWithDepth called")
        backgroundHandler?.post {
            try {
                Log.d("TCP", "backgroundHandler 실행")
                // 1. 카메라 프레임 (JPEG 압축)
                val jpegStream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, jpegStream)
                val jpegBytes = jpegStream.toByteArray()

                // 2. Raw depth 데이터 (float → byte)
                val depthArray = classifier.runRawDepth(bitmap) // float[]
                val depthBuffer = ByteBuffer
                    .allocate(depthArray.size * 4)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                depthBuffer.asFloatBuffer().put(depthArray)
                val depthBytes = depthBuffer.array()

                // 3. zlib 압축
                val compressedDepth = compressBytes(depthBytes)

                // 4. 송신 패킷 구조: [JPEG길이][Depth길이][JPEG데이터][Depth데이터]
                val header = ByteBuffer.allocate(8)
                    .putInt(jpegBytes.size)
                    .putInt(compressedDepth.size)
                    .array()

                outputStream?.apply {
                    write(header)
                    write(jpegBytes)
                    write(compressedDepth)
                    flush()

                    // ✅ 전송 로그
                    Log.d("TCP", "Frame sent: JPEG=${jpegBytes.size} bytes, Depth=${compressedDepth.size} bytes")
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    protected fun sendFrameWithDepthAndPose(
        bitmap: Bitmap,
        depthArray: FloatArray,
        yaw: Float
    ) {
        backgroundHandler?.post {
            try {
                val NET_ORDER = java.nio.ByteOrder.BIG_ENDIAN

                // 1) JPEG
                val jpegStream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, jpegStream)
                val jpegBytes = jpegStream.toByteArray()

                // 2) depth float[] → byte[]
                val depthBuffer = ByteBuffer
                    .allocate(depthArray.size * 4)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                depthBuffer.asFloatBuffer().put(depthArray)
                val depthBytes = depthBuffer.array()

                // 3) header + yaw
                val header = ByteBuffer.allocate(12)
                    .order(NET_ORDER)
                    .putInt(jpegBytes.size)
                    .putInt(depthBytes.size)
                    .putFloat(yaw)      // ★ yaw 포함
                    .array()

                outputStream?.let { out ->
                    out.write(header)
                    out.write(jpegBytes)
                    out.write(depthBytes)
                    out.flush()
                } ?: Log.w("TCP", "outputStream is null – frame dropped")

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    /*protected fun sendFrameWithDepthNoCompression(classifier: DepthClassifier, bitmap: Bitmap) {
        Log.d("TCP", "sendFrameWithDepthNoCompression called")
        backgroundHandler?.post {
            try {
                Log.d("TCP", "backgroundHandler 실행 (압축 없이 전송)")

                // 1. 카메라 프레임 (JPEG 압축)
                val jpegStream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, jpegStream)
                val jpegBytes = jpegStream.toByteArray()

                // 2. Raw depth 데이터 (float → byte, little endian)
                val depthArray = classifier.runRawDepth(bitmap) // float[]
                val depthBuffer = ByteBuffer
                    .allocate(depthArray.size * 4)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                depthBuffer.asFloatBuffer().put(depthArray)
                val depthBytes = depthBuffer.array() // ✅ 그대로 전송

                // 3. 송신 패킷 구조: [JPEG길이][Depth길이][JPEG데이터][Depth데이터]
                val header = ByteBuffer.allocate(8)
                    .putInt(jpegBytes.size)
                    .putInt(depthBytes.size)
                    .array()

                outputStream?.apply {
                    write(header)
                    write(jpegBytes)
                    write(depthBytes)
                    flush() // 매 프레임마다 flush 가능, 필요하면 나중에 주기적으로 변경 가능

                    Log.d("TCP", "Frame sent (no compression): JPEG=${jpegBytes.size} bytes, Depth=${depthBytes.size} bytes")
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }*/

    protected fun sendFrameWithDepthNoCompression(bitmap: Bitmap, depthArray: FloatArray) {
        Log.d("TCP", "sendFrameWithDepthNoCompression called (depth len=${depthArray.size})")

        backgroundHandler?.post {
            try {
                val NET_ORDER = java.nio.ByteOrder.BIG_ENDIAN

                // ----------------------------------------------------
                // 1) JPEG 생성
                // ----------------------------------------------------
                val jpegStream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, jpegStream)
                val jpegBytes = jpegStream.toByteArray()

                // ----------------------------------------------------
                // 2) depth float[] -> byte[]
                // ----------------------------------------------------
                val depthBuffer = ByteBuffer
                    .allocate(depthArray.size * 4)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN) // float 저장은 LITTLE_ENDIAN 유지
                depthBuffer.asFloatBuffer().put(depthArray)
                val depthBytes = depthBuffer.array()

                // ----------------------------------------------------
                // 3) header (8 bytes): [jpegLen][depthLen]
                //    ※ 네트워크 전송은 BIG_ENDIAN 사용
                // ----------------------------------------------------
                val header = ByteBuffer.allocate(1 + 8)
                    .order(NET_ORDER)
                    .put(0x01)                  // ★ frame packet
                    .putInt(jpegBytes.size)
                    .putInt(depthBytes.size)
                    .array()

                // ----------------------------------------------------
                // 4) 전송
                // ----------------------------------------------------
                outputStream?.let { out ->
                    out.write(header)
                    out.write(jpegBytes)
                    out.write(depthBytes)
                    out.flush()

                    Log.d(
                        "TCP",
                        "Frame sent: JPEG=${jpegBytes.size} bytes, Depth=${depthBytes.size} bytes"
                    )

                } ?: run {
                    Log.w("TCP", "outputStream is null – frame dropped")
                }

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private val sensorListener = object : SensorEventListener {

        override fun onSensorChanged(event: SensorEvent) {

            val timestamp = event.timestamp  // ns

            when (event.sensor.type) {

                Sensor.TYPE_GYROSCOPE -> {
                    val gyroX = event.values[0]
                    val gyroY = event.values[1]
                    val gyroZ = event.values[2]

                    Log.d("GyroTest", "X:$gyroX Y:$gyroY Z:$gyroZ ts:$timestamp")

                    // ⚠️ 기존 유지 (네가 yaw로 쓰는 축)
                    sendGyro(gyroY, timestamp)
                }

                Sensor.TYPE_LINEAR_ACCELERATION -> {
                    val accX = event.values[0]
                    val accY = event.values[1]
                    val accZ = event.values[2]

                    Log.d("AccelTest", "X:$accX Y:$accY Z:$accZ ts:$timestamp")

                    sendAccel(accX, accY, accZ, timestamp)
                }
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }


    private fun sendGyro(gyroZ: Float, timestamp: Long) {
        backgroundHandler?.post {
            try {
                val buffer = ByteBuffer
                    .allocate(1 + 8 + 4)
                    .order(ByteOrder.BIG_ENDIAN)
                    .put(0x02)
                    .putLong(timestamp)
                    .putFloat(gyroZ)
                    .array()

                outputStream?.let {
                    it.write(buffer)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun sendAccel(ax: Float, ay: Float, az: Float, timestamp: Long) {
        backgroundHandler?.post {
            try {
                val buffer = ByteBuffer
                    .allocate(1 + 8 + 4 * 3)
                    .order(ByteOrder.BIG_ENDIAN)
                    .put(0x03)          // ★ accel packet
                    .putLong(timestamp)
                    .putFloat(ax)
                    .putFloat(ay)
                    .putFloat(az)
                    .array()

                outputStream?.let {
                    it.write(buffer)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }


    private fun compressBytes(data: ByteArray): ByteArray {
        Log.d("COMPRESS", "압축 전 크기: ${data.size} bytes")
        val deflater = Deflater(1)
        deflater.setInput(data)
        deflater.finish()
        val output = ByteArray(data.size)
        val compressedSize = deflater.deflate(output)
        val compressedData = output.copyOf(compressedSize)
        Log.d("COMPRESS", "압축 후 크기: $compressedSize bytes")
        Log.d("COMPRESS", "압축률: ${"%.2f".format(compressedSize.toFloat() / data.size * 100)}%")

        // 🔹 압축된 데이터 일부를 hex로 출력 (앞부분만)
        val sampleSize = minOf(64, compressedData.size)
        val hexPreview = compressedData
            .take(sampleSize)
            .toByteArray()
            .toList()
            .joinToString(" ") { "%02X".format(it) }

        Log.d("COMPRESS", "압축된 데이터 샘플 (${sampleSize}B): $hexPreview")
        return output.copyOf(compressedSize)
    }

    // CameraActivity 내부에 추가
    protected fun sendTestPattern(width: Int = 256, height: Int = 256) {
        backgroundHandler?.post {
            try {
                val W = width
                val H = height
                val size = W * H
                // 1) 테스트 패턴: float 값 0,1,2,...
                val testArray = FloatArray(size) { i -> i.toFloat() }

                // 2) ByteBuffer에 little endian으로 넣기
                val depthBuffer = ByteBuffer
                    .allocate(size * 4)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                depthBuffer.asFloatBuffer().put(testArray)
                val depthBytes = depthBuffer.array()

                // 3) 간단 헤더(여기서는 JPEG 길이 0으로 표시하고 depth만 보냄)
                val header = ByteBuffer.allocate(8)
                    .order(java.nio.ByteOrder.BIG_ENDIAN) // 기존 헤더와 동일하게 네트워크 바이트 오더
                    .putInt(0)        // jpeg len = 0 (없음)
                    .putInt(depthBytes.size)
                    .array()

                outputStream?.apply {
                    write(header)
                    // no jpeg
                    write(depthBytes)
                    flush()
                }

                // 로그: 처음 10개의 float을 hex로 찍어보기
                val sample = depthBytes.take(40).toByteArray() // 첫 10 float = 40 bytes
                val hex = sample.joinToString(" ") { "%02X".format(it) }
                Log.d("TEST_SEND", "Sent test pattern. depthBytes.size=${depthBytes.size}, sampleHex=$hex")

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }


}
