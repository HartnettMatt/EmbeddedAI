# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := linux
TARGET_ARCH := x86_64

# Default compiler and tool names:
TOOLCHAIN:=gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

#TFLM_ROOT := /home/erke9581/embeddedai-new/tflite-micro/tensorflow/lite/micro
TFLM_ROOT := /home/matt/Documents/Education/Graduate/EmbeddedAI/tflite-micro

#THIRD_PARTY := $(TFLM_ROOT)/tools/make/downloads
THIRD_PARTY := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads


INCLUDES := -I$(TFLM_ROOT) \
  -I. \
  -I$(THIRD_PARTY)/flatbuffers/include \
  -I$(THIRD_PARTY)/gemmlowp \
  -I$(THIRD_PARTY)/kissfft



CC_WARNINGS := \
  -Wno-sign-compare \
  -Wno-double-promotion \
  -Wno-unused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

COMMON_FLAGS := \
  -Werror \
  -fno-unwind-tables \
  -fno-asynchronous-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(CC_WARNINGS) \
  -DTF_LITE_USE_CTIME

CXXFLAGS := \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  $(COMMON_FLAGS)


LDFLAGS := -lm\
  -L$(TFLM_ROOT)/gen/$(TARGET)_$(TARGET_ARCH)_default_gcc/lib/ -ltensorflow-microlite\
  -Wl,--fatal-warnings -Wl,--gc-sections 

#%.o:	%.cc
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


#micro_speec_test: micro_speech_test.o
#  $(CXX) $(CXXFLAGS) $(INCLUDES) -o micro_speech_test micro_speech_test.o $(LDFLAGS)


MICRO_SPEECH_COMMON_SRCS := \
audio_processor.cc \
micro_speech_processor.cc \
models/audio_preprocessor_float_model_data.cc \
models/audio_preprocessor_int8_model_data.cc \
models/micro_speech_quantized_model_data.cc \
models/matt_micro_speech_quantized_model_data.cc \
testdata/matt_down_1000ms_audio_data.cc \
testdata/matt_go_1000ms_audio_data.cc \
testdata/matt_left_1000ms_audio_data.cc \
testdata/matt_no_1000ms_audio_data.cc \
testdata/matt_right_1000ms_audio_data.cc \
testdata/matt_silence_1000ms_audio_data.cc \
testdata/matt_stop_1000ms_audio_data.cc \
testdata/matt_up_1000ms_audio_data.cc \
testdata/matt_yes_1000ms_audio_data.cc \
testdata/no_1000ms_audio_data.cc \
testdata/no_30ms_audio_data.cc \
testdata/noise_1000ms_audio_data.cc \
testdata/silence_1000ms_audio_data.cc \
testdata/yes_1000ms_audio_data.cc \
testdata/yes_30ms_audio_data.cc


MICRO_SPEECH_TEST_SRCS := \
$(MICRO_SPEECH_COMMON_SRCS) \
micro_speech_test.cc


MICRO_SPEECH_SRCS := \
$(MICRO_SPEECH_COMMON_SRCS) \
main_functions.cc \
audio_provider_modl.cc \
main.cc


MICRO_SPEECH_COMMON_OBJS := \
audio_processor.o \
micro_speech_processor.o \
models/audio_preprocessor_float_model_data.o \
models/audio_preprocessor_int8_model_data.o \
models/micro_speech_quantized_model_data.o \
models/matt_micro_speech_quantized_model_data.o \
testdata/matt_down_1000ms_audio_data.o \
testdata/matt_go_1000ms_audio_data.o \
testdata/matt_left_1000ms_audio_data.o \
testdata/matt_no_1000ms_audio_data.o \
testdata/matt_right_1000ms_audio_data.o \
testdata/matt_silence_1000ms_audio_data.o \
testdata/matt_stop_1000ms_audio_data.o \
testdata/matt_up_1000ms_audio_data.o \
testdata/matt_yes_1000ms_audio_data.o \
testdata/no_1000ms_audio_data.o \
testdata/no_30ms_audio_data.o \
testdata/noise_1000ms_audio_data.o \
testdata/silence_1000ms_audio_data.o \
testdata/yes_1000ms_audio_data.o \
testdata/yes_30ms_audio_data.o

MICRO_SPEECH_TEST_OBJS := \
$(MICRO_SPEECH_COMMON_OBJS) \
micro_speech_test.o 


MICRO_SPEECH_OBJS := \
$(MICRO_SPEECH_COMMON_OBJS) \
main_functions.o \
audio_provider_mock.o \
main.o


#$(MICRO_SPEECH_SRCS:.cc=.o)

%.o:   %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

all:	micro_speech_test micro_speech


all_micro_speech_test_objs := $(MICRO_SPEECH_TEST_OBJS) 

all_micro_speech_objs := $(MICRO_SPEECH_OBJS)

micro_speech_test:       $(all_micro_speech_test_objs)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o micro_speech_test $(all_micro_speech_test_objs) $(LDFLAGS)

micro_speech:       $(all_micro_speech_objs)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o micro_speech $(all_micro_speech_objs) $(LDFLAGS)


#micro_speech.o:	$(MICRO_SPEECH_SRCS)
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(MICRO_SPEECH_SRCS) -o micro_speech.o



#micro_speech:	$(MICRO_SPEECH_SRCS)
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -v -o micro_speech $(LDFLAGS) $(MICRO_SPEECH_SRCS) 


clean:
	$(RM) *.o *.d *.map models/*.o testdata/*.o micro_speech micro_speech_test
