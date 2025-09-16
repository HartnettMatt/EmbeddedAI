#ifndef AUDIO_PROVIDER_H_
#define AUDIO_PROVIDER_H_

#include <cstdint>
#include "tensorflow/lite/c/common.h"

int AudioProviderNumSamplesAvailable();
TfLiteStatus AudioProviderGetSamples(int16_t* output_audio_samples, int num_samples_to_get, int advance_by);
TfLiteStatus AudioProviderInitialize();
int AudioProviderGetCurrentSampleTime();

#endif