package org.pytorch.KWS;

import android.content.Context;
// import android.graphics.Bitmap;
// import android.graphics.BitmapFactory;
// import android.media.AudioFormat;
// import android.media.AudioRecord;
// import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
// import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Button;
import android.view.View;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
// import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
  private Button mBtn1;
  private Button mBtn2;
  /*private  AudioRecord mAudioRecord;
  private boolean isRecording = false;
  int mic = MediaRecorder.AudioSource.MIC;
  int sr = 16000;
  // int SAMPLE_DURATION_MS = 5000;
  int channel = AudioFormat.CHANNEL_IN_MONO;
  int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
  int bufferSize = 2 * AudioRecord.getMinBufferSize(sr, channel, audioFormat);
  int RECORDING_LENGTH = sr;
  int recordingOffset = 0;
  short[] recordingBuffer = new short[RECORDING_LENGTH];*/
  Module module_encoder = null;
  Module module_decoder = null;
  String audio1;
  String audio2;
  String className;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    // init();
    try {
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      audio1 = assetFilePath(this, "left.wav");
      audio2 = assetFilePath(this, "test.wav");
      module_encoder = Module.load(assetFilePath(this, "encoder.pt"));
      module_decoder = Module.load(assetFilePath(this, "decoder.pt"));
    } catch (Exception e) {
      Log.e("KWS", "Error reading assets", e);
      finish();
    }
    mBtn1 = findViewById(R.id.button);
    mBtn1.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v){
        long start = System.currentTimeMillis( );
        float[] mfccInput = new float[5040];
        try {
          WavFile wavFile = WavFile.openWavFile(new File(audio1));
          int numChannels = wavFile.getNumChannels();
          double[] buffer = new double[16000 * numChannels];
          int framesRead;
          do {
            // Read frames into buffer
            framesRead = wavFile.readFrames(buffer, 16000);
            MFCC mfccConvert = new MFCC();
            mfccInput = mfccConvert.process(buffer);
          }
          while (framesRead != 0);
        } catch (Exception e) {
          Log.e("KWS", "Error reading assets", e);
          finish();
        }
        long mid = System.currentTimeMillis( );
        className = getKeyword(mfccInput, module_encoder, module_decoder);
        long end = System.currentTimeMillis( );
        long diff_feature = mid - start;
        long diff_model = end - mid;
        String diff_time_feature = Long.toString(diff_feature);
        String diff_time_model = Long.toString(diff_model);
        TextView textView = findViewById(R.id.text);
        textView.setText(className + ", feature_time: " + diff_time_feature + "ms" + ", model_time: " + diff_time_model + "ms");
      }
    });

    mBtn2 = findViewById(R.id.button2);
    mBtn2.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v){
        long start = System.currentTimeMillis( );
        float[] mfccInput = new float[5040];
        try{
          WavFile wavFile = WavFile.openWavFile(new File(audio2));
          int numChannels = wavFile.getNumChannels();
          double[] buffer = new double[16000 * numChannels];
          int framesRead;
          do {
            // Read frames into buffer
            framesRead = wavFile.readFrames(buffer, 16000);
            MFCC mfccConvert = new MFCC();
            mfccInput = mfccConvert.process(buffer);
          }
          while (framesRead != 0);
        } catch (Exception e) {
          Log.e("KWS", "Error reading assets", e);
          finish();
        }
        long mid = System.currentTimeMillis( );
        className = getKeyword(mfccInput, module_encoder, module_decoder);
        long end = System.currentTimeMillis( );
        long diff_feature = mid - start;
        long diff_model = end - mid;
        String diff_time_feature = Long.toString(diff_feature);
        String diff_time_model = Long.toString(diff_model);
        TextView textView = findViewById(R.id.text);
        textView.setText(className + ", feature_time: " + diff_time_feature + "ms" + ", model_time: " + diff_time_model + "ms");
      }
    });
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  public String getKeyword(float[] mfccInput, Module module_encoder, Module module_decoder){
    String className;
    final Tensor eInputTensor;
    eInputTensor = Tensor.fromBlob(mfccInput, new long[]{1, 126, 40});
    // encoder
    final Tensor encoderTensor = module_encoder.forward(IValue.from(eInputTensor)).toTensor();

    // transpose encoder tensor
    final float[] encoderBuffer = encoderTensor.getDataAsFloatArray();
    float[] encoderBuffer_trans = new float[42*32];
    for(int i = 0; i < encoderBuffer.length; i++) {
      int a = i / 42;
      int b = i % 42;
      int idx = a+32*b;
      encoderBuffer_trans[idx] = encoderBuffer[i];
    }

    // preparing decoder input tensor
    final Tensor dInputTensor;
    dInputTensor = Tensor.fromBlob(encoderBuffer_trans, new long[]{1, 42, 32});
    // decoder
    final Tensor outputTensor = module_decoder.forward(IValue.from(dInputTensor)).toTensor();
    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }
    className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
    return className;
  }
}
