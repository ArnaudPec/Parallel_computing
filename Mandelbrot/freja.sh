#!/bin/bash

make clean
freja compile;
 freja run test ;
 ./drawplots.r ;
make
eog *.svg mandelbrot.ppm
