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

TfLiteStatus LoadMicroSpeechModelAndPerformInference(
    const MyMicroSpeech::Features& features, const char* expected_label) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);
  TF_LITE_MICRO_EXPECT(model->version() == TFLITE_SCHEMA_VERSION);
  TF_LITE_MICRO_CHECK_FAIL();

  MicroSpeechOpResolver op_resolver;
  TF_LITE_MICRO_EXPECT(RegisterOps(op_resolver) == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();

  tflite::MicroInterpreter interpreter(model, op_resolver, g_micro_speech_arena, kMicroSpeechArenaSize);

  TF_LITE_MICRO_EXPECT(interpreter.AllocateTensors() == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();

  MicroPrintf("MicroSpeech model arena size = %u",
              interpreter.arena_used_bytes());

  TfLiteTensor* input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_CHECK_FAIL();
  // check input shape is compatible with our feature data size
  TF_LITE_MICRO_EXPECT_EQ(kFeatureElementCount,
                          input->dims->data[input->dims->size - 1]);
  TF_LITE_MICRO_CHECK_FAIL();

  TfLiteTensor* output = interpreter.output(0);
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
  TF_LITE_MICRO_EXPECT(interpreter.Invoke() == kTfLiteOk);
  TF_LITE_MICRO_CHECK_FAIL();

  // Dequantize output values
  float category_predictions[kCategoryCount];
  MicroPrintf("MicroSpeech category predictions for <%s>", expected_label);
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
  TF_LITE_MICRO_EXPECT_STRING_EQ(expected_label,
                                 kCategoryLabels[prediction_index]);
  TF_LITE_MICRO_CHECK_FAIL();

  return kTfLiteOk;
}
