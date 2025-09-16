#include <algorithm>
#include <cstdint>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "micro_model_settings.h"
#include "models/micro_speech_quantized_model_data.h"




namespace {

tflite::MicroInterpreter *g_speech_interpreter = nullptr;

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kMicroSpeechArenaSize = 28584;  // xtensa p6
alignas(16) uint8_t g_micro_speech_arena[kMicroSpeechArenaSize];


using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;

TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

}


TfLiteStatus MicroSpeechModelInitialize() {

  static const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);
  TF_LITE_MICRO_EXPECT(model->version() == TFLITE_SCHEMA_VERSION);
  TF_LITE_MICRO_CHECK_FAIL();

  static MicroSpeechOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT(RegisterOps(op_resolver) == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();


  g_speech_interpreter = new tflite::MicroInterpreter(model, op_resolver, g_micro_speech_arena, kMicroSpeechArenaSize);

  TF_LITE_MICRO_EXPECT(g_speech_interpreter->AllocateTensors() == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();

  MicroPrintf("MicroSpeech model arena size = %u",
              g_speech_interpreter->arena_used_bytes());

  return kTfLiteOk;

}


TfLiteStatus MicroSpeechModelInference(
    const MyMicroSpeech::Features& features, int* predicted_output_index) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  MicroPrintf("MicroSpeechModelInference: features[0][0]=%d, features[0][1]=%d, features[0][2]=%d", 
                                          features[0][0], features[0][1], features[0][2]);
  MicroPrintf("MicroSpeechModelInference: features[10][0]=%d, features[10][1]=%d, features[10][2]=%d", 
                                          features[10][0], features[10][1], features[10][2]);
  MicroPrintf("MicroSpeechModelInference: features[40][0]=%d, features[40][1]=%d, features[40][2]=%d", 
                                          features[40][0], features[40][1], features[40][2]);

  
  TfLiteTensor* input = g_speech_interpreter->input(0);
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_CHECK_FAIL();
  // check input shape is compatible with our feature data size
  TF_LITE_MICRO_EXPECT_EQ(kFeatureElementCount,
                          input->dims->data[input->dims->size - 1]);
  TF_LITE_MICRO_CHECK_FAIL();

  TfLiteTensor* output = g_speech_interpreter->output(0);
  TF_LITE_MICRO_EXPECT(output != nullptr);
  TF_LITE_MICRO_CHECK_FAIL();
  // check output shape is compatible with our number of prediction categories
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount,
                          output->dims->data[output->dims->size - 1]);
  TF_LITE_MICRO_CHECK_FAIL();

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  std::copy_n(&features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(input));
  TF_LITE_MICRO_EXPECT(g_speech_interpreter->Invoke() == kTfLiteOk);

  TF_LITE_MICRO_CHECK_FAIL();

  // Dequantize output values
  float category_predictions[kCategoryCount];
//  MicroPrintf("MicroSpeech category predictions for <%s>", expected_label);
  MicroPrintf("MicroSpeech category predictions:");
  for (int i = 0; i < kCategoryCount; i++) {
    category_predictions[i] =
        (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
        output_scale;
    MicroPrintf("  %.4f %s", static_cast<double>(category_predictions[i]),
                kCategoryLabels[i]);
  }
  int prediction_index =
      std::distance(std::begin(category_predictions),
                    std::max_element(std::begin(category_predictions),
                                     std::end(category_predictions)));
//  TF_LITE_MICRO_EXPECT_STRING_EQ(expected_label,
//                                 kCategoryLabels[prediction_index]);
//  TF_LITE_MICRO_CHECK_FAIL();
  *predicted_output_index = prediction_index;
  g_speech_interpreter->Reset();
  
  return kTfLiteOk;
}
