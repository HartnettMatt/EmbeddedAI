#ifndef AUDIO_PROCESSOR_H_
#define AUDIO_PROCESSOR_H_

#include <cstdint>

#include "tensorflow/lite/core/c/common.h"

#include "micro_model_settings.h"


TfLiteStatus AudioPreprocessorInitialize();

TfLiteStatus AudioPreprocessorInference(const int16_t* audio_data,
                              const size_t audio_data_size,
                              MyMicroSpeech::Features* features_output, bool reset_interpreter = false);

#endif  // AUDIO_PROCESSOR_H_
