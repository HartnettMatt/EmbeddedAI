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