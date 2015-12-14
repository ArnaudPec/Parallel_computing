hello_world:
	#gcc hello_world_cl.c CLutilities.c -lOpenCL -o hello_world_cl #this is just for testing at home
	gcc hello_world_cl.c CLutilities.c -lOpenCL -I/sw/cuda/4.2/include -o hello_world_cl

image_filter:
	gcc CLutilities.c filter.c readppm.c milli.c -lGL -lglut -lOpenCL -I/sw/cuda/4.2/include -o filter
