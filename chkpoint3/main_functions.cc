#include "main_functions.h"
#include "micro_model_settings.h"
#include "audio_provider.h"
#include "audio_processor.h"
#include "micro_speech_processor.h"

#include "tensorflow/lite/micro/micro_log.h"

namespace {
  constexpr int num_samples_for_inference = kAudioSampleFrequency;  // 1 second of audio (16000 samples)
  int16_t input_audio[num_samples_for_inference]; // 1 second of audio (16000 samples)


}

void setup() {

  AudioProviderInitialize();
  AudioPreprocessorInitialize();
  MicroSpeechModelInitialize();


}

void loop() {
    // Inference was trained on spectrograms of 49 frames (each 30ms, with a stride of 20ms, for a total of 1s) of 40 feature buckets.  
    // So, we'll loop until we have enough samples to run inference (16000 samples, which is the kAudioSampleFrequency constant).
    // We could overlap the inference, e.g., 1) 0-1s, 2) 0.5s-1.5s, 3) 1s-2s, etc. 
    //    But, we  

  // check if 1s of samples are available
  int num_samples_available = AudioProviderNumSamplesAvailable();

  if (num_samples_available < num_samples_for_inference) {
    MicroPrintf("Not enough samples available: %d, need %d", num_samples_available, num_samples_for_inference);
    return; // not enough data yet
  }

  // Get the samples from the audio provider
  TfLiteStatus status = AudioProviderGetSamples(input_audio, num_samples_for_inference, num_samples_for_inference*0.25); //num_samples_for_inference); 
  if (status != kTfLiteOk) {
    MicroPrintf("Failed to get audio samples");
  }
  MicroPrintf("Got %d samples from audio provider at time %d ms", num_samples_for_inference, AudioProviderGetCurrentSampleTime());

  // Run the audio preprocessor to convert the raw audio into features
  MyMicroSpeech::Features features;
  status = AudioPreprocessorInference(input_audio, num_samples_for_inference, &features);
  if (status != kTfLiteOk) {
    MicroPrintf("Failed to preprocess audio samples");
    return; // exit if preprocessing fails
  }

  // Run the model on the features to get a prediction
  int predicted_output_index = -1;
  status = MicroSpeechModelInference(features, &predicted_output_index, false); // false is don't print_debug
  if (status != kTfLiteOk) {
    MicroPrintf("Failed to run model inference");
  }

  // Output the prediction (in real application, you might want to do something with this result)
  const char* predicted_label = "unknown";
  if (predicted_output_index >=0 && predicted_output_index < kCategoryCount) {
    predicted_label = kCategoryLabels[predicted_output_index];
  }
  MicroPrintf("Predicted label: %s", predicted_label);

}