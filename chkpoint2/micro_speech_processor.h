#ifndef MICRO_SPEECH_PROCESSOR_H_
#define MICRO_SPEECH_PROCESSOR_H_

#include <cstdint>

#include "tensorflow/lite/core/c/common.h"

#include "micro_model_settings.h"

TfLiteStatus MicroSpeechModelInitialize();


TfLiteStatus MicroSpeechModelInference(
    const MyMicroSpeech::Features& features, int* predicted_output_index);

#endif  // MICRO_SPEECH_PROCESSOR_H_
