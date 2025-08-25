// Skeleton code for hello_world.cc with snippets at the bottom to cut and paste


 
////  SNIP 1 (Include necessary headers)


using namespace std;

namespace {

// for our test case, we'll keep track of how many times we perform inference
int inference_count = 0;
constexpr int kInferencesPerCycle = 200; // for ~6.6s on Arduino Nano 33 BLE Sense

// This constant represents the range of x values our model was trained on,
// which is from 0 to (2 * Pi). We approximate Pi to avoid requiring additional
// libraries.
const float kXrange = 2.f * 3.14159265359f;



//// SNIP 2  (define some globals used for the model and interpreter)


//// SNIP 3  (Define a custom operator resolver for the Hello World example.)

}  // namespace




void setup() {

  TfLiteStatus status; // for return values;

  // Structure:
     // 1. Initialize the target system.
     // 2. Map the model into a usable data structure.
     // 3. Create an operator resolver.
     // 4. Allocate tensors.
     // 5. Get input and output tensors.


  //// SNIP 4 (Initialize the target system.)
  
  //// SNIP 5 (Map the model into a usable data structure.)

  //// SNIP 6 (Create an operator resolver.)

  //// SNIP 7 (Allocate tensors.)

  //// SNIP 8 (Get input and output tensors.)



} // setup


// For dev, we just want to print out the values
// For running on Arduino, we'd have a different implementation which calulates a brightness value and sets it on an LED
void HandleOutput(float x_value, float y_value) {
  // Log values 
  MicroPrintf("x_value=%f, y_value=%f\n", (double)x_value, (double)y_value);
}

void loop() {

 // General Structure:
  // 1. Set the input tensor with a value to feed into the model.   
  // 2. Run inference.
  // 3. Obtain the output from the model and do something with it.

  //// SNIP 9 (set the input tensor with a value to feed into the model.)

  //// SNIP 10 (Run inference.)

  //// SNIP 11 (Obtain the output from the model and do something with it.)



  // For our test case, we will increment the inference_count and reset it (this is not a general thing you'd do)
  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
 
} 


// Arduino wouldn't have this main function, but we do for testing
int main(int argc, char* argv[]) {

    setup();
    while(true){ 
        loop();
        sleep(1);
    }

}



////////////////////////////////////////////////



  //// BEGIN SNIP 9

  // 1. Set the input tensor with a value to feed into the model.   

  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // using float version:
  input->data.f[0] = x;

  // if using int8 version:
  // Quantize the input from floating-point to integer
  //int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  //input->data.int8[0] = x_quantized;


  //// END SNIP 9



  //// BEGIN SNIP 11

  //  3. Obtain the output from the model and do something with it.

  // using float:
  float y = output->data.f[0];

  // if using int8:
  // Obtain the quantized output from model's output tensor
  //int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  //float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(x, y);
  //// END SNIP 11




//// BEGIN SNIP 1

#include <math.h>
#include <unistd.h> // For sleep()

// For Arduino, use:
//#include <Chirale_TensorFlowLite.h>

// could use int8 or float version, we'll use float
//#include "hello_world_int8_model_data.h"
#include "hello_world_float_model_data.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//// END SNIP 1


  //// BEGIN SNIP 10

  // 2. Run inference.   

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
    return;
  }

  //// END SNIP 10



//// BEGIN SNIP 2

// some globals used 
const tflite::Model* model = nullptr;  // the model
tflite::MicroInterpreter* interpreter = nullptr; // the interpreter (what can perform inference)
TfLiteTensor* input = nullptr; // the input tensor (where we put the input data)
TfLiteTensor* output = nullptr; // the output tensor (where we get the output data)
float output_scale = 0.0f; // for quantized models, the scale used to convert the output data
int output_zero_point = 0; // for quantized models, the zero point used to convert the output data

// the tensor_arena is the memory used for tensors (inputs, outputs, and intermediate tensors)
  // Determining the Arena size is a bit of trial/error. 
  // Docs say The exact arena usage can be determined  using the RecordingMicroInterpreter.
constexpr int kTensorArenaSize = 2000;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // Keep aligned to 16 bytes for CMSIS

//// END SNIP 2





  //// BEGIN SNIP 5

  // 2. Map the model into a usable data structure.

  // using float:
  model = tflite::GetModel(g_hello_world_float_model_data);
  // if using int8
  // model = tflite::GetModel(g_hello_world_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  //// END SNIP 5


  //// BEGIN SNIP 4 

  // 1. Initialize the target system.

  // The tflite::InitializeTarget function, invoked during startup, is actually a user-implemented function. 
  // It serves as a call to the system_setup.cc file, which users are expected to implement as part of integrating 
  // TensorFlow Lite for Microcontrollers. This function provides a location for the initialization of hardware 
  // drivers, such as UART, SPI, etc.
  // For Arduino, see:
  // https://github.com/spaziochirale/Chirale_TensorFlowLite/blob/main/src/tensorflow/lite/micro/system_setup.cpp

  tflite::InitializeTarget();

  //// END SNIP 4



  //// BEGIN SNIP 6

  // 3. Create an operator resolver.

  
  static HelloWorldOpResolver op_resolver;  // static means the allocation of memory survives the function call
  status = RegisterOps(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("RegisterOps failed with status: %d\n", status);
    return;
  }

  static tflite::MicroInterpreter micro_interpreter(model, op_resolver, tensor_arena,
                                           kTensorArenaSize);
  interpreter = &micro_interpreter; // Set the interpreter global variable


  //// END SNIP 6


//// BEGIN SNIP 3

// Define a custom operator resolver for the Hello World example.
// This resolver only needs one operator, the FullyConnected operator 
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;
TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;
}

//// END SNIP 3


  //// BEGIN SNIP 7

  // 4. Allocate tensors.
  // This step is necessary to allocate memory for the input, output, and intermediate tensors.
  // The AllocateTensors() function will allocate memory for the tensors based on the model's
  // input and output shapes, and the tensor arena size.
  // It will also set the input and output tensors in the interpreter.
  status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed with status: %d\n", status);
    return;
  }


  //// END SNIP 7


  //// BEGIN SNIP 8

  // 5. Get input and output tensors.

  input = interpreter->input(0);
  if (input == nullptr) {
    MicroPrintf("Input tensor is null\n");
    return;
  }

  output = interpreter->output(0);
  if (output == nullptr) {
    MicroPrintf("Output tensor is null\n");
    return;
  }

  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;


  //// END SNIP 8
