package com.example.theultimaterecognizer

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.MediaStore.Images.Media.getBitmap
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView

import androidx.appcompat.app.AppCompatActivity
import com.example.theultimaterecognizer.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var bitmap: Bitmap
    private lateinit var imageview:ImageView

    @SuppressLint("CutPasteId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imageview=findViewById(R.id.image)
        val tv:TextView=findViewById(R.id.textView)
        val select:Button= findViewById(R.id.button2)
        select.setOnClickListener(View.OnClickListener {
            val intent= Intent(Intent.ACTION_GET_CONTENT)
            intent.type="image/*"
            startActivityForResult(intent,100)
        })
        val predict:Button=findViewById(R.id.button)
        predict.setOnClickListener(View.OnClickListener {
            val resized:Bitmap=Bitmap.createScaledBitmap(bitmap,224,224,true)
            val model = MobilenetV110224Quant.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)


            val tbuffer=TensorImage.fromBitmap(resized)
            val byteBuffer=tbuffer.buffer
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            tv.setText(outputFeature0.floatArray[10].toString())

// Releases model resources if no longer used.
            model.close()

        })
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        imageview.setImageURI(data?.data)
        val uri: Uri?=data?.data
        bitmap=getBitmap(this.contentResolver,uri)
    }
}