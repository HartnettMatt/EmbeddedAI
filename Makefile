# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := linux
TARGET_ARCH := x86_64

# Default compiler and tool names:
TOOLCHAIN:=gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar


TFLM_ROOT := /home/matt/Documents/Education/Graduate/EmbeddedAI/tflite-micro
BINDIR := bin



THIRD_PARTY := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads

INCLUDES := -I$(TFLM_ROOT) \
  -I. \
  -I$(THIRD_PARTY)/flatbuffers/include \
  -I$(THIRD_PARTY)/gemmlowp



CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
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
  $(TFLM_ROOT)/gen/$(TARGET)_$(TARGET_ARCH)_default_gcc/lib/libtensorflow-microlite.a\
  -Wl,--fatal-warnings -Wl,--gc-sections 

%.o:	%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

all:	$(BINDIR)/hello_world

all_objs := hello_world.o hello_world_float_model_data.o
# if using int8 hello_world_int8_model_data.o

$(BINDIR)/hello_world:	$(all_objs)
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BINDIR)/hello_world $(all_objs) $(LDFLAGS)


clean:
	$(RM) *.o *.d *.map $(BINDIR)/hello_world
