# Lab 2 Report
### Matt Hartnett, ECEN 4003, Embedded AI

## Part 1 Notes:

I decided to run things locally, that way I don't have to worry about co-lab interrupting things (and I think it's interesting to see .
the performance on a machine I am familiar with).

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

Looking at the the confusion map, a lot of it makes intuitive sense: I can understand why "go" and "no" are often confused for each other, but some are a little less obvious, like "down and "go"
