/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "micro_model_settings.h"
#include "audio_processor.h"
#include "micro_speech_processor.h"

#include "testdata/no_1000ms_audio_data.h"
#include "testdata/no_30ms_audio_data.h"
#include "testdata/noise_1000ms_audio_data.h"
#include "testdata/silence_1000ms_audio_data.h"
#include "testdata/yes_1000ms_audio_data.h"
#include "testdata/yes_30ms_audio_data.h"


namespace {


MyMicroSpeech::Features g_features;



TfLiteStatus TestAudioSample(const char* label, const int16_t* audio_data,
                             const size_t audio_data_size) {
  TF_LITE_ENSURE_STATUS(
      GenerateFeatures(audio_data, audio_data_size, &g_features));
  TF_LITE_ENSURE_STATUS(
      LoadMicroSpeechModelAndPerformInference(g_features, label));
  return kTfLiteOk;
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(NoFeatureTest) {
  int8_t expected_feature[kFeatureSize] = {
      126, 103, 124, 102, 124, 102, 123, 100, 118, 97, 118, 100, 118, 98,
      121, 100, 121, 98,  117, 91,  96,  74,  54,  87, 100, 87,  109, 92,
      91,  80,  64,  55,  83,  74,  74,  78,  114, 95, 101, 81,
  };

  TF_LITE_ENSURE_STATUS(GenerateFeatures(
      g_no_30ms_audio_data, g_no_30ms_audio_data_size, &g_features));
  for (size_t i = 0; i < kFeatureSize; i++) {
    TF_LITE_MICRO_EXPECT_EQ(g_features[0][i], expected_feature[i]);
    TF_LITE_MICRO_CHECK_FAIL();
  }
}

TF_LITE_MICRO_TEST(YesFeatureTest) {
  int8_t expected_feature[kFeatureSize] = {
      124, 105, 126, 103, 125, 101, 123, 100, 116, 98,  115, 97,  113, 90,
      91,  82,  104, 96,  117, 97,  121, 103, 126, 101, 125, 104, 126, 104,
      125, 101, 116, 90,  81,  74,  80,  71,  83,  76,  82,  71,
  };

  TF_LITE_ENSURE_STATUS(GenerateFeatures(
      g_yes_30ms_audio_data, g_yes_30ms_audio_data_size, &g_features));
  for (size_t i = 0; i < kFeatureSize; i++) {
    TF_LITE_MICRO_EXPECT_EQ(g_features[0][i], expected_feature[i]);
    TF_LITE_MICRO_CHECK_FAIL();
  }
}

TF_LITE_MICRO_TEST(NoTest) {
  TestAudioSample("no", g_no_1000ms_audio_data, g_no_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(YesTest) {
  TestAudioSample("yes", g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(SilenceTest) {
  TestAudioSample("silence", g_silence_1000ms_audio_data,
                  g_silence_1000ms_audio_data_size);
}

TF_LITE_MICRO_TEST(NoiseTest) {
  TestAudioSample("silence", g_noise_1000ms_audio_data,
                  g_noise_1000ms_audio_data_size);
}

TF_LITE_MICRO_TESTS_END
