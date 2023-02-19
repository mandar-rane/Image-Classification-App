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
import org.tensorflow.lite.DataType
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

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, fa)
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



        val model = Model.newInstance(applicationContext)

// Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32

        val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(imageSize * imageSize)
        image!!.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0

        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                var value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16)).and(0xFF).toFloat() * ((1F) / 1))
                byteBuffer.putFloat(((value shr 8)).and(0xFF).toFloat() * ((1F) / 1))
                byteBuffer.putFloat((value).and(0xFF).toFloat() * ((1F) / 1))
            }
        }


        inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val confidences = outputFeature0.floatArray
        val conf = arrayListOf<Float>()
        confidences.forEach {
            conf.add(it)
        }
        Log.d("confindeces: ","${conf.toString()}")
        var maxPos = 0
        for (i in 1 until confidences.size) {
            if (confidences[i] > confidences[maxPos]) {
                maxPos = i
            }
        }
        Log.d("confindeces: ","${maxPos.toString()}")

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


// Releases model resources if no longer used.
        model.close()

    }


}