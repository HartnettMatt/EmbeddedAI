#include <cstdint>
#include <cstddef>
#include <algorithm>

#include "tensorflow/lite/core/c/common.h"

#include "testdata/no_1000ms_audio_data.h"
#include "testdata/no_30ms_audio_data.h"
#include "testdata/noise_1000ms_audio_data.h"
#include "testdata/silence_1000ms_audio_data.h"
#include "testdata/yes_1000ms_audio_data.h"
#include "testdata/yes_30ms_audio_data.h"

#include "micro_model_settings.h"
#include "audio_provider.h"

namespace {
// Make a buffer big enough to hold each of the 4 audio data arrays (yes, no, noise, silence -- each at 1sec, or 16000 samples)
constexpr int32_t audio_data_buffer_size = g_yes_1000ms_audio_data_size+g_no_1000ms_audio_data_size+
                          g_noise_1000ms_audio_data_size+g_silence_1000ms_audio_data_size;
int16_t audio_data_buffer[audio_data_buffer_size] = {0};
int32_t audio_data_samples_available = 0;
int32_t audio_data_curr_rd_ix = 0;
int32_t total_samples_read_all_time = 0; // good for debugging.
// For real application (e.g., Arduino, etc.), we'd also have a write index 
}  // namespace


TfLiteStatus AudioProviderInitialize() {
   // Copy the audio from separate arrays to a global array
   int start_sample = 0;

   std::copy_n(g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size,  audio_data_buffer + start_sample);
   start_sample += g_yes_1000ms_audio_data_size;

   std::copy_n(g_no_1000ms_audio_data, g_no_1000ms_audio_data_size,  audio_data_buffer + start_sample);
   start_sample += g_no_1000ms_audio_data_size;

   std::copy_n(g_noise_1000ms_audio_data, g_noise_1000ms_audio_data_size,  audio_data_buffer + start_sample);
   start_sample += g_noise_1000ms_audio_data_size;

   std::copy_n(g_silence_1000ms_audio_data, g_silence_1000ms_audio_data_size,  audio_data_buffer + start_sample);
   start_sample += g_silence_1000ms_audio_data_size;
 
   audio_data_samples_available = start_sample;
   audio_data_curr_rd_ix = 0;

   return kTfLiteOk;
}

// will get the num_samples_to_get (returns kTfLiteSuccess) or nothing (returns kTfLiteFail) 
// advance_by is the number of samples to advance the read index by.  
//     You may want it to be 0 if you want to peek at the data.
//     You may want it to be the same as num_samples to get if you want to not get these samples again 
//     You may want to advance it by a smaller number if you want to overlap some samples (in case a words is split across two calls) 
TfLiteStatus AudioProviderGetSamples(int16_t* output_audio_samples, int num_samples_to_get, int advance_by) {
    if (num_samples_to_get > audio_data_samples_available) {
        return kTfLiteError;
    }

    // Need to handle conditions where num_samples_to_get is larger than the remaining samples before buffer circles 
    int32_t samples_until_wrap = audio_data_buffer_size - audio_data_curr_rd_ix;
    if (num_samples_to_get > samples_until_wrap) {
        // Copy the first part of the samples until wrap
        int16_t* tmp_ptr = audio_data_buffer + audio_data_curr_rd_ix;
        std::copy_n(tmp_ptr, samples_until_wrap, output_audio_samples);
        // Copy the rest of the samples from the start of the buffer
        std::copy_n(audio_data_buffer, num_samples_to_get - samples_until_wrap, output_audio_samples + samples_until_wrap);
        audio_data_curr_rd_ix = num_samples_to_get - samples_until_wrap;
    } else {
        // Copy the requested number of samples from the current index
        int16_t* tmp_ptr = audio_data_buffer + audio_data_curr_rd_ix;
        std::copy_n(tmp_ptr, num_samples_to_get, output_audio_samples);
        audio_data_curr_rd_ix += num_samples_to_get;
    }
    total_samples_read_all_time += num_samples_to_get;


    // Version 1: don't adjust audio_data_samples_available as we'll mock that data is always available 
    // Version 2: could adjust audio_data_samples_available as samples are read (so, minus num_samples_to_get)
    return kTfLiteOk;
}


// Version 1 returns the variable audio_data_samples_available, which doesn't change
// Version 2 could extend to each time called, add some number of samples available  
// In real application (on Arduino, etc.) there would be a callback function that 
//      would update the number of samples available as it gets read from the device 
int AudioProviderNumSamplesAvailable() {
    return audio_data_samples_available;
}


int AudioProviderGetCurrentSampleTime() {
  // if total_samples_read_all_time is 16000, I want it to return 1000 ms 
  // if total_samples_read_all_time is 480, I want it to return 30ms
  return total_samples_read_all_time / (kAudioSampleFrequency/1000); // return the current sample time in seconds  

}
