
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "micro_model_settings.h"
#include "audio_processor.h"
#include "models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test.h"


namespace {

tflite::MicroInterpreter *g_audio_interpreter = nullptr;
const tflite::Model* g_audio_model = nullptr;

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kAudioArenaSize = 28584;  // xtensa p6
alignas(16) uint8_t g_audio_arena[kAudioArenaSize];



using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

TfLiteStatus RegisterOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}
}

TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output) {
  TfLiteTensor* input = g_audio_interpreter->input(0);
  if (input == nullptr) { 
    MicroPrintf("GenerateSingleFeature: input tensor is null"); 
    return kTfLiteError; 
  }

  // check input shape is compatible with our audio sample size
  if (kAudioSampleDurationCount != audio_data_size)  { 
    MicroPrintf("GenerateSingleFeature: kAudioSampleDurationCount != audio_data_size"); 
    return kTfLiteError; 
  }

  if (kAudioSampleDurationCount !=  input->dims->data[input->dims->size - 1]) {
    MicroPrintf("GenerateSingleFeature: kAudioSampleDurationCount (%d) != input->dims->data[input->dims->size - 1] (%d)",
                kAudioSampleDurationCount, input->dims->data[input->dims->size - 1]);    
    return kTfLiteError; 
  }


  TfLiteTensor* output = g_audio_interpreter->output(0);
  if (output == nullptr) {
    MicroPrintf("GenerateSingleFeature: output tensor is null"); return kTfLiteError; }

    // check output shape is compatible with our feature size
  if (kFeatureSize !=   output->dims->data[output->dims->size - 1]) {
    MicroPrintf("GenerateSingleFeature: kFeatureSize (%d) !=   output->dims->data[output->dims->size - 1] (%d )",
                kFeatureSize, output->dims->data[output->dims->size - 1]); return kTfLiteError; 
  }

  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));
  TfLiteStatus status = g_audio_interpreter->Invoke();
  if (status != kTfLiteOk) {
    MicroPrintf("GenerateSingleFeature: Invoke failed with status %d", status);
    return status;
  }

  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
              feature_output);

//  g_audio_interpreter->Reset();
  return kTfLiteOk;
}



TfLiteStatus AudioPreprocessorInitialize() {

  g_audio_model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);
  if (g_audio_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("AudioPreprocessorInitialize: Model version mismatch");
    return kTfLiteError;
  }


  static AudioPreprocessorOpResolver op_resolver;
  TfLiteStatus status = RegisterOps(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("AudioPreprocessorInitialize: RegisterOps failed with status %d", status);
    return status;
  }

  g_audio_interpreter = new tflite::MicroInterpreter(g_audio_model, op_resolver, g_audio_arena, kAudioArenaSize);

  status = g_audio_interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("AudioPreprocessorInitialize: AllocateTensors failed with status %d", status);
    return status;
  }

  MicroPrintf("AudioPreprocessor model arena size = %u",
              g_audio_interpreter->arena_used_bytes());

  return kTfLiteOk;
}

TfLiteStatus AudioPreprocessorInference(const int16_t* audio_data,
                              const size_t audio_data_size,
                              MyMicroSpeech::Features* features_output, bool reset_interpreter) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.


  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
    // MicroPrintf("AudioPreprocessorInference: feature_index=%d", feature_index);
    // MicroPrintf("AudioPreprocessorInference: audio_data[0]=%d, audio_data[1]=%d, audio_data[3]=%d", audio_data[0], audio_data[1], audio_data[3]);
    TfLiteStatus status = 
        GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index]);
    if (status != kTfLiteOk) {
      MicroPrintf("AudioPreprocessorInference: GenerateSingleFeature failed with status %d", status);
    }
    //MicroPrintf("AudioPreprocessorInference: features_output[%d][0]=%d, features_output[%d][1]=%d, features_output[%d][2]=%d", 
    //                                      feature_index, (*features_output)[feature_index][0], 
    //                                      feature_index, (*features_output)[feature_index][1], 
    //                                      feature_index, (*features_output)[feature_index][2]);

    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }
  g_audio_interpreter->Reset();

  return kTfLiteOk;
}

