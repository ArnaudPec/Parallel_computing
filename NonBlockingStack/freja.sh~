#!/bin/bash

# Just comment line 4 and uncomment line 5 in the multicore room
#freja="/home/arnaud/lib_zib/home/TDDD56/usr/bin/freja"
freja="/home/daniel/LiU/tddd56/freja/home/TDDD56/usr/bin/freja"
#freja="freja"

make clean
$freja compile;
$freja run test ;
 ./drawplots.r ;
make
#eog *.svg
feh --magick-timeout 1 *.svg
