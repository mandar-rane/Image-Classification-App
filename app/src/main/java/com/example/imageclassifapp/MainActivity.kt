package com.example.imageclassifapp

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ThumbnailUtils
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.graphics.get
import com.example.imageclassifapp.databinding.ActivityMainBinding
import com.example.imageclassifapp.ml.Model
import com.example.imageclassifapp.ml.Newmodel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {


    private val CAMERA_PERMISSION_CODE = 1
    private val CAMERA_REQ_CODE = 2
    private val imageSize = 224
    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.takePictureBtn.setOnClickListener {
            if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent: Intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, CAMERA_REQ_CODE)
            } else {
                requestPermissions(
                    arrayOf(android.Manifest.permission.CAMERA),
                    CAMERA_PERMISSION_CODE
                )
            }
        }

        binding.selectPictureBtn.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent,1)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == CAMERA_REQ_CODE) {
                var image: Bitmap = data!!.extras!!.get("data") as Bitmap
                val dimension = Math.min(image.width, image.height)
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                binding.image.setImageBitmap(image)

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                classifyImage(image)
            }else{
                val dat: Uri? = data!!.data
                var image: Bitmap? = null
                try {
                    image = MediaStore.Images.Media.getBitmap(this.contentResolver,dat)
                }catch (e: IOException){
                    e.printStackTrace()
                }
                binding.image.setImageBitmap(image)
                image = Bitmap.createScaledBitmap(image!!, imageSize, imageSize, false)
                classifyImage(image)
            }
        }
    }

    private fun classifyImage(image: Bitmap?) {
        //
        val model = Model.newInstance(applicationContext)
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        //

        val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(4*imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(imageSize * imageSize)
        image!!.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val value = intValues[pixel++]
                val r = (value shr 16) and 0xFF
                val g = (value shr 8) and 0xFF
                val b = value and 0xFF
                // Normalize pixel values to [0, 1]
                val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
                byteBuffer.putFloat(normalizedPixelValue)
            }
        }

        //
        inputFeature0.loadBuffer(byteBuffer)
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        //

        val confidences = outputFeature0.floatArray

        var maxPos = 0
        for (i in 1 until confidences.size) {
            if (confidences[i] > confidences[maxPos]) {
                maxPos = i
            }
        }

        val classes = arrayOf(
            "Bloody_mary",
            "Cosmopolitan",
            "Espresso_martini",
            "Margarita",
            "Mimosa",
            "Mojito",
            "Moscow_mule",
            "old_fashioned"
        )

        binding.result.text = classes[maxPos]

        model.close()

    }


}