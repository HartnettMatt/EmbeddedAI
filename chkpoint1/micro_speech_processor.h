#ifndef MICRO_SPEECH_PROCESSOR_H_
#define MICRO_SPEECH_PROCESSOR_H_

#include <cstdint>

#include "tensorflow/lite/core/c/common.h"

#include "micro_model_settings.h"

TfLiteStatus LoadMicroSpeechModelAndPerformInference(
    const MyMicroSpeech::Features& features, const char* expected_label);

#endif  // MICRO_SPEECH_PROCESSOR_H_
