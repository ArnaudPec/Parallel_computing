#!/bin/bash

# Just comment line 4 and uncomment line 5 in the multicore room
freja="/home/arnaud/lib_zib/home/TDDD56/usr/bin/freja"
#freja="freja"

make clean
$freja compile;
$freja run test ;
 ./drawplots.r ;
make
eog *.svg mandelbrot.ppm
