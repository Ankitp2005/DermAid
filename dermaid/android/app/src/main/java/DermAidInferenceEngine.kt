package com.dermaid

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class DermAidResult(
    val conditionCode: String,
    val conditionProbs: List<Float>,
    val severityTier: String,
    val severityProbs: List<Float>,
    val confidence: Float,
    val inferenceTimeMs: Long
)

class DermAidInferenceEngine(context: Context) {

    private var interpreter: Interpreter? = null
    
    private val conditionLabels = arrayOf("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
    private val severityLabels = arrayOf("Low Risk", "Refer Soon", "Refer Immediately")

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
        }
        val modelBuffer = loadModelFile(context, "dermaid_int8.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        
        // 1 batch * 224 height * 224 width * 3 channels * 1 byte for INT8
        val byteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(224 * 224)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        
        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val value = intValues[pixel++]
                
                // Subtract 128 to convert 0..255 unsigned into -128..127 signed (INT8 offset)
                val r = ((value shr 16 and 0xFF) - 128).toByte()
                val g = ((value shr 8 and 0xFF) - 128).toByte()
                val b = ((value and 0xFF) - 128).toByte()
                
                byteBuffer.put(r)
                byteBuffer.put(g)
                byteBuffer.put(b)
            }
        }
        return byteBuffer
    }

    fun classify(bitmap: Bitmap): DermAidResult {
        val byteBuffer = preprocessBitmap(bitmap)
        
        // Allocate output arrays. Assuming TF Lite automatically dequantizes to Float arrays
        // based on the model signature. If the model outputs INT8, these would need to be ByteArray
        val conditionOut = arrayOf(FloatArray(7))
        val severityOut = arrayOf(FloatArray(3))
        val confidenceOut = arrayOf(FloatArray(1))
        
        val inputs = arrayOf<Any>(byteBuffer)
        val outputs = mutableMapOf<Int, Any>(
            0 to conditionOut,
            1 to severityOut,
            2 to confidenceOut
        )
        
        val startTime = System.currentTimeMillis()
        interpreter?.runForMultipleInputsOutputs(inputs, outputs)
        val endTime = System.currentTimeMillis()
        val inferenceTime = endTime - startTime
        
        val condArr = conditionOut[0]
        val sevArr = severityOut[0]
        val confVal = confidenceOut[0][0]
        
        // Find argmax
        var condIdx = 0
        var maxCondValue = condArr[0]
        for (i in 1 until condArr.size) {
            if (condArr[i] > maxCondValue) {
                maxCondValue = condArr[i]
                condIdx = i
            }
        }
        
        var sevIdx = 0
        var maxSevValue = sevArr[0]
        for (i in 1 until sevArr.size) {
            if (sevArr[i] > maxSevValue) {
                maxSevValue = sevArr[i]
                sevIdx = i
            }
        }
        
        return DermAidResult(
            conditionCode = conditionLabels[condIdx],
            conditionProbs = condArr.toList(),
            severityTier = severityLabels[sevIdx],
            severityProbs = sevArr.toList(),
            confidence = confVal,
            inferenceTimeMs = inferenceTime
        )
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
