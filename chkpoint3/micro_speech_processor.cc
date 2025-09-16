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

#include "tensorflow/lite/micro/testing/micro_test.h"



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
  if (model == nullptr) {
    MicroPrintf("MicroSpeechModelInitialize: Failed to get model from data");
    return kTfLiteError;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("MicroSpeech model schema version mismatch: expected %d, got %d",
                TFLITE_SCHEMA_VERSION, model->version());
    return kTfLiteError;
  }

  static MicroSpeechOpResolver op_resolver;
  TfLiteStatus status = RegisterOps(op_resolver); 
  if (status != kTfLiteOk) {
    MicroPrintf("MicroSpeechModelInitialize: RegisterOps failed with status %d", status);
    return status;
  }


  g_speech_interpreter = new tflite::MicroInterpreter(model, op_resolver, g_micro_speech_arena, kMicroSpeechArenaSize);
  if (g_speech_interpreter == nullptr) {
    MicroPrintf("MicroSpeechModelInitialize: Failed to create MicroInterpreter");
    return kTfLiteError;
  }

  status = g_speech_interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("MicroSpeechModelInitialize: AllocateTensors failed with status %d", status);
    return status;
  }

  

  MicroPrintf("MicroSpeech model arena size = %u",
              g_speech_interpreter->arena_used_bytes());

  return kTfLiteOk;

}


TfLiteStatus MicroSpeechModelInference(
    const MyMicroSpeech::Features& features, int* predicted_output_index, bool print_debug) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  //MicroPrintf("MicroSpeechModelInference: features[0][0]=%d, features[0][1]=%d, features[0][2]=%d", 
  //                                        features[0][0], features[0][1], features[0][2]);
  //MicroPrintf("MicroSpeechModelInference: features[10][0]=%d, features[10][1]=%d, features[10][2]=%d", 
  //                                        features[10][0], features[10][1], features[10][2]);
  //MicroPrintf("MicroSpeechModelInference: features[40][0]=%d, features[40][1]=%d, features[40][2]=%d", 
  //                                        features[40][0], features[40][1], features[40][2]);

  
  TfLiteTensor* input = g_speech_interpreter->input(0);
  if (input == nullptr) {
    MicroPrintf("MicroSpeechModelInference: input tensor is null");
    return kTfLiteError;
  }
  // check input shape is compatible with our feature data size
  if (kFeatureElementCount != input->dims->data[input->dims->size - 1]) {
    MicroPrintf("MicroSpeechModelInference: kFeatureElementCount (%d) != input->dims->data[input->dims->size - 1] (%d)",
                kFeatureElementCount, input->dims->data[input->dims->size - 1]);
    return kTfLiteError;
  }

  // Get the output tensor

  TfLiteTensor* output = g_speech_interpreter->output(0);
  if (output == nullptr) {
    MicroPrintf("MicroSpeechModelInference: output tensor is null");
    return kTfLiteError;
  }


  // check output shape is compatible with our number of prediction categories
  if (kCategoryCount != output->dims->data[output->dims->size - 1]) {
    MicroPrintf("MicroSpeechModelInference: kCategoryCount (%d) != output->dims->data[output->dims->size - 1] (%d)",
                kCategoryCount, output->dims->data[output->dims->size - 1]);
    return kTfLiteError;
  }

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  std::copy_n(&features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(input));
  TfLiteStatus status = g_speech_interpreter->Invoke();
  if (status != kTfLiteOk) {
    MicroPrintf("MicroSpeechModelInference: Invoke failed with status %d", status);
    return status;
  }

  float category_predictions[kCategoryCount];
  // Dequantize output values
  if (print_debug) {
     MicroPrintf("MicroSpeech category predictions:");
  }
  for (int i = 0; i < kCategoryCount; i++) {
    category_predictions[i] =
         (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
         output_scale;
    if (print_debug) {
        MicroPrintf("  %.4f %s", static_cast<double>(category_predictions[i]),
                   kCategoryLabels[i]);
    }
  }
  int prediction_index =
      std::distance(std::begin(category_predictions),
                    std::max_element(std::begin(category_predictions),
                                     std::end(category_predictions)));
  *predicted_output_index = prediction_index;
  g_speech_interpreter->Reset();
  
  return kTfLiteOk;
}
