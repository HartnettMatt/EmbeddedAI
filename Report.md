# Lab 2 Report
### Matt Hartnett, ECEN 4003, Embedded AI

## Part 1 Notes:

I decided to run things locally, that way I don't have to worry about co-lab interrupting things (and I think it's interesting to see the performance on a machine I am familiar with).

In order to run the python, you can either upload it to co-lab, or run it locally using the virtual environment created and managed with `uv`.
Install `uv` and run `uv sync` on the repo in order to create the virtual environment. 
After that, use either `uv run python ./train/part1.py` or `source .venv/bin/activate && python ./train/part1.py`.


I went through most of the example, but then stopped to check my understanding once I got to graphing the spectrograms.

This is how I think of spectrograms:
Imagine breaking your signal into a bunch of little time slices, and each one of those time slices can overlap, but they're all mapped across the x axis.
Then take the FFT of each slice, and the frequency dimension becomes the y axis.
The spectrogram is the top down view of all of the FFTs with color representing the power of the signal's frequency for that slice.

More slices = better time resolution, worse frequency resolution (larger fft bins, more spectral leakage)
Windowing: can help with spectral leakage (I'm a blackman-harris fan, personally)

Looking at the the confusion map, a lot of it makes intuitive sense: I can understand why "go" and "no" are often confused for each other, but some are a little less obvious, like "down and "go".

On the exporting side, I found that the export section of the online guide was basically entirely broken due to some confusing type errors that I didn't think was worth the effort to fix.
Instead, I just wrote my own export that created a `matt_micro_speech_qunatized.tflite` file.

From there, I edited the `Makefile` to match my tflite-micro installation locatioin, then I ran the generate.sh script to get the *.h and *.cc files.
Once those files existed, I included them in the Makefile in much the same way as the current models.

After that, I changed the include statement in the `micro_speech_processor.cc` file to include my file instead of the default.
I also had to change the `tflite::GetModel()` call.

I built my code and ran it, but I got the following error:

``` 
AudioPreprocessor model arena size = 9944
Didn't find op for builtin opcode 'RESIZE_BILINEAR'
Failed to get registration from op code RESIZE_BILINEAR
 
MicroSpeechModelInitialize: AllocateTensors failed with status 1
Got 16000 samples from audio provider at time 1000 ms
Segmentation fault (core dumped)
```

Here are the various fixes I needed to implement to get everything to work:
- Change model input size to not resize, but use the default
- Add a bunch of OpCodes to the `op_resolver`
- Increase the size of the `op_resolver` to 10 instead of 4
- Increase the arena size
- Change the training data to flatten it
- Change the kCategory count and labels

# Part 2 Notes:
After recording all of my data, I needed to change the sample rate since I recorded in too high of a sample rate, which stopped the model from being able to process the data. 

I used ChatGPT 5-Thinking with the following prompt to create the `resample_wavs.py` script.

Prompt:
```
Write a simple python script that iterates through a directory, finds all files of type .wav, then converts those .wav files from a 44.1kHz sample rate to a 16kHz sample rate without losing any information.
```

I then ran the output as follows:
`python resample_wavs.py ./testdata/testdata/ --inplace`

It worked great, the audio was downsampled to 16kHz with minimal hassle.

I found that the model was very bad at predicting what the words were, so I modified it to improve it's accuracy. Here are the things I did:
- Increase epoch count to 30 and disabled auto stopping
- Increased the internal model shape to be input -> 128 -> 256 -> 128 -> output instead of input -> 256 -> 128 -> output
- Messed around with dropout to find a good balance between training loss and validation loss

I don't know why, but my model is really bad at accurately predicting my "left" audio file. I know it's the correct file, I've listend to it to double check, but it's still consistently predicting it incorrectly.

In order to use the audio files in my c++ application, I had to add them to `audio_provider_mock.cc` in three places:
```
#include "testdata/matt_down_1000ms_audio_data.h"
#include "testdata/matt_go_1000ms_audio_data.h"
...
```

```
constexpr int32_t audio_data_buffer_size =
    g_matt_down_1000ms_audio_data_size +
    g_matt_go_1000ms_audio_data_size +
    g_matt_left_1000ms_audio_data_size +
    g_matt_no_1000ms_audio_data_size +
    g_matt_right_1000ms_audio_data_size +
    g_matt_silence_1000ms_audio_data_size +
    g_matt_stop_1000ms_audio_data_size +
    g_matt_up_1000ms_audio_data_size +
    g_matt_yes_1000ms_audio_data_size;
```


```
std::copy_n(g_matt_down_1000ms_audio_data, g_matt_down_1000ms_audio_data_size, audio_data_buffer + start_sample);
start_sample += g_matt_down_1000ms_audio_data_size;

std::copy_n(g_matt_go_1000ms_audio_data, g_matt_go_1000ms_audio_data_size, audio_data_buffer + start_sample);
start_sample += g_matt_go_1000ms_audio_data_size;
...
```

I also had to include those header files in my `Makefile`.