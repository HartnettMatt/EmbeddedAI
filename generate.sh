#!/bin/bash


BASEDIR=/home/erke9581/embeddedai-new/tflite-micro/

cd testdata
python3 $BASEDIR/tensorflow/lite/micro/tools/generate_cc_arrays.py testdata *.wav
mv testdata/* .
cd ../models
python3 $BASEDIR/tensorflow/lite/micro/tools/generate_cc_arrays.py models *.tflite
mv models/* .
cd ..

