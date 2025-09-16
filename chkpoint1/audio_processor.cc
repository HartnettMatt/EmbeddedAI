
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "micro_model_settings.h"
#include "audio_processor.h"
#include "models/audio_preprocessor_int8_model_data.h"


namespace {

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
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_CHECK_FAIL();
  // check input shape is compatible with our audio sample size
  TF_LITE_MICRO_EXPECT_EQ(kAudioSampleDurationCount, audio_data_size);
  TF_LITE_MICRO_CHECK_FAIL();
  TF_LITE_MICRO_EXPECT_EQ(kAudioSampleDurationCount,
                          input->dims->data[input->dims->size - 1]);
  TF_LITE_MICRO_CHECK_FAIL();

  TfLiteTensor* output = interpreter->output(0);
  TF_LITE_MICRO_EXPECT(output != nullptr);
  TF_LITE_MICRO_CHECK_FAIL();
  // check output shape is compatible with our feature size
  TF_LITE_MICRO_EXPECT_EQ(kFeatureSize,
                          output->dims->data[output->dims->size - 1]);
  TF_LITE_MICRO_CHECK_FAIL();

  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));
  TF_LITE_MICRO_EXPECT(interpreter->Invoke() == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();
  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
              feature_output);

  return kTfLiteOk;
}

//int8_t[kFeatureCount][kFeatureSize]

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              MyMicroSpeech::Features* features_output) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);
  TF_LITE_MICRO_EXPECT(model->version() == TFLITE_SCHEMA_VERSION);
  TF_LITE_MICRO_CHECK_FAIL();

  AudioPreprocessorOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT(RegisterOps(op_resolver) == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();

  tflite::MicroInterpreter interpreter(model, op_resolver, g_audio_arena, kAudioArenaSize);

  TF_LITE_MICRO_EXPECT(interpreter.AllocateTensors() == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();

  MicroPrintf("AudioPreprocessor model arena size = %u",
              interpreter.arena_used_bytes());

  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
    TF_LITE_ENSURE_STATUS(
        GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index], &interpreter));
    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  return kTfLiteOk;
}
