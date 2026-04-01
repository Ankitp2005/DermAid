package com.dermaid

import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.dermaid.databinding.ActivityMainBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Assumption: CaseLoggerRepository is implemented elsewhere in the project
class CaseLoggerRepository {
    fun logResult(condition: String, severity: String, confidence: Float) {
        // SQLite logging logic
    }
}

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var inferenceEngine: DermAidInferenceEngine
    private val caseLogger = CaseLoggerRepository()
    
    private var currentLanguage = "en"
    private var lastResult: DermAidResult? = null

    // English Matrix
    private val referralMatrixEn = mapOf(
        Pair("nv", "Low Risk") to Triple("No Action Needed", "This appears to be a common mole...monitor for changes in size, color, shape. Re-visit if lesion grows beyond 6mm.", "GREEN"),
        Pair("df", "Low Risk") to Triple("Monitor Regularly", "Likely a benign skin growth. Not dangerous. Recommend annual skin check.", "GREEN"),
        Pair("bkl", "Low Risk") to Triple("Monitor — No Rush", "Benign skin lesion. Not cancerous. Avoid sun exposure. Follow-up in 3 months if change observed.", "GREEN"),
        Pair("vasc", "Refer Soon") to Triple("Refer Within 1 Week", "Vascular lesion. Fill referral form, send to PHC doctor within 7 days. Do not apply pressure.", "YELLOW"),
        Pair("akiec", "Refer Soon") to Triple("Refer Within 3 Days — Pre-cancerous", "Possible pre-cancerous lesion. FILL REFERRAL SLIP NOW. Schedule district hospital appointment.", "YELLOW"),
        Pair("bcc", "Refer Immediately") to Triple("URGENT — Refer Within 24 Hours", "Possible Basal Cell Carcinoma. Complete emergency referral form. District hospital within 24 hours.", "RED"),
        Pair("mel", "Refer Immediately") to Triple("EMERGENCY — Refer TODAY", "High-risk melanoma signal. Medical emergency. Call district hospital NOW. Transport patient today.", "RED")
    )

    // Hindi Matrix
    private val referralMatrixHi = mapOf(
        Pair("nv", "Low Risk") to Triple("कोई कार्रवाई आवश्यक नहीं", "यह एक सामान्य तिल प्रतीत होता है... इसके आकार, रंग और रूप में बदलाव की निगरानी करें। यदि घाव 6 मिमी से बड़ा हो जाता है तो पुनः दिखाएं।", "GREEN"),
        Pair("df", "Low Risk") to Triple("नियमित निगरानी करें", "यह एक सौम्य त्वचा की वृद्धि है। खतरनाक नहीं है। वार्षिक त्वचा जांच की सिफारिश की जाती है।", "GREEN"),
        Pair("bkl", "Low Risk") to Triple("निगरानी करें — कोई जल्दी नहीं", "सौम्य त्वचा का घाव। कैंसर रहित है। धूप से बचें। यदि कोई बदलाव दिखाई दे तो 3 महीने में फॉलो-अप करें।", "GREEN"),
        Pair("vasc", "Refer Soon") to Triple("1 सप्ताह के भीतर रेफर करें", "संवहनी घाव। रेफरल फॉर्म भरें, 7 दिनों के भीतर पीएचसी डॉक्टर के पास भेजें। दबाव लागू न करें।", "YELLOW"),
        Pair("akiec", "Refer Soon") to Triple("3 दिन के भीतर रेफर करें — पूर्व कैंसर", "संभावित पूर्व कैंसर घाव। अभी रेफरल पर्ची भरें। जिला अस्पताल में अपॉइंटमेंट लें।", "YELLOW"),
        Pair("bcc", "Refer Immediately") to Triple("अति आवश्यक — 24 घंटे के भीतर रेफर करें", "संभावित बेसल सेल कार्सिनोमा। आपातकालीन रेफरल फॉर्म पूरा करें। 24 घंटे के भीतर जिला अस्पताल रेफर करें।", "RED"),
        Pair("mel", "Refer Immediately") to Triple("आपातकाल — आज ही रेफर करें", "उच्च जोखिम वाले मेलेनोमा का संकेत। चिकित्सा आपातकाल। जिला अस्पताल को अभी बुलाएं। मरीज को आज ही अस्पताल पहुंचाएं।", "RED")
    )

    private val takePicturePreview = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        bitmap?.let { processImage(it) }
    }

    private val pickGalleryImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
            processImage(bitmap)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        inferenceEngine = DermAidInferenceEngine(this)

        binding.btnCamera.setOnClickListener {
            takePicturePreview.launch(null)
        }

        binding.btnGallery.setOnClickListener {
            pickGalleryImage.launch("image/*")
        }
        
        binding.btnToggleLanguage.setOnClickListener {
            currentLanguage = if (currentLanguage == "en") "hi" else "en"
            binding.btnToggleLanguage.text = if (currentLanguage == "en") "HI" else "EN"
            lastResult?.let { updateUI(it) }
        }
        
        binding.btnRetake.setOnClickListener {
            binding.referralCard.visibility = View.GONE
            takePicturePreview.launch(null)
        }
        
        binding.btnFillForm.setOnClickListener {
            // Placeholder: Start form activity
        }
    }

    private fun processImage(bitmap: Bitmap) {
        binding.progressBar.visibility = View.VISIBLE
        binding.referralCard.visibility = View.GONE
        binding.imageView.setImageBitmap(bitmap)

        lifecycleScope.launch {
            val isQualityGood = checkImageQuality(bitmap)
            if (!isQualityGood) {
                binding.progressBar.visibility = View.GONE
                Snackbar.make(binding.root, "Image too blurry/dark. Please retake.", Snackbar.LENGTH_LONG).show()
                return@launch
            }

            val result = withContext(Dispatchers.Default) {
                inferenceEngine.classify(bitmap)
            }
            
            lastResult = result
            
            // Log to local DB on background thread
            withContext(Dispatchers.IO) {
                caseLogger.logResult(result.conditionCode, result.severityTier, result.confidence)
            }
            
            updateUI(result)
            binding.progressBar.visibility = View.GONE
        }
    }

    private fun updateUI(result: DermAidResult) {
        val matrix = if (currentLanguage == "en") referralMatrixEn else referralMatrixHi
        val key = Pair(result.conditionCode, result.severityTier)
        
        var (actionTitle, instruction, urgencyColor) = matrix[key] ?: Triple("Unknown", "No specific instruction.", "GRAY")

        // Critical safety threshold override
        if (result.confidence < 0.60f) {
            actionTitle = if (currentLanguage == "en") "Refer Soon — Low Confidence Result" else "जल्द रेफर करें - कम विश्वास परिणाम"
            urgencyColor = "YELLOW"
        }

        val bgColor = when (urgencyColor) {
            "GREEN" -> Color.parseColor("#4CAF50")
            "YELLOW" -> Color.parseColor("#FFC107")
            "RED" -> Color.parseColor("#F44336")
            else -> Color.LTGRAY
        }

        binding.referralCard.visibility = View.VISIBLE
        binding.referralCard.setCardBackgroundColor(bgColor)
        binding.tvActionTitle.text = actionTitle
        binding.tvInstruction.text = instruction
        binding.tvConfidence.text = "Confidence: %.1f%%".format(result.confidence * 100)
    }

    /**
     * Replicates check_image_quality in Kotlin.
     * Calculates Brightness via mean luma, Blur via discrete Laplacian variance.
     */
    private suspend fun checkImageQuality(bitmap: Bitmap): Boolean = withContext(Dispatchers.Default) {
        // Downscale to speed up the convolution
        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val width = scaled.width
        val height = scaled.height
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        
        var sumLuma = 0.0
        val lumaValues = DoubleArray(pixels.size)
        
        for (i in pixels.indices) {
            val color = pixels[i]
            val r = (color shr 16) and 0xFF
            val g = (color shr 8) and 0xFF
            val b = color and 0xFF
            val luma = 0.299 * r + 0.587 * g + 0.114 * b
            lumaValues[i] = luma
            sumLuma += luma
        }
        
        val meanLuma = sumLuma / pixels.size
        if (meanLuma < 40.0) return@withContext false // Reject dark images
        
        var laplacianSqSum = 0.0
        var varianceCount = 0
        
        // Simplified Laplacian kernel
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val idx = y * width + x
                var lVal = 0.0
                
                lVal += lumaValues[idx - width]      // Top
                lVal += lumaValues[idx - 1]          // Left
                lVal += lumaValues[idx] * -4.0       // Center
                lVal += lumaValues[idx + 1]          // Right
                lVal += lumaValues[idx + width]      // Bottom
                
                laplacianSqSum += lVal * lVal
                varianceCount++
            }
        }
        
        val variance = laplacianSqSum / varianceCount
        // Reject blurry images (100.0 is an arbitrary empirical threshold)
        if (variance < 100.0) return@withContext false 
        
        return@withContext true
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceEngine.close()
    }
}
