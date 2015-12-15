/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2
/*#define LOCAL false */
/*#if LOCAL //that is run algorithm using local memory*/
__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{ 
    unsigned int i = get_global_id(1) % 512;
    unsigned int j = get_global_id(0) % 512;

    //size of local image
    unsigned const local_n = 20;
    unsigned const local_m = 20;

    unsigned local_i = get_local_id(1);//i + wg_i * WORK_GROUP_DIM;
    unsigned local_j = get_local_id(0);//i + wg_i * WORK_GROUP_DIM;

    __local unsigned char local_image[20*20*3]; // will contain part of image to be blurred and necessary surrounding pixels
    unsigned local_v = (local_i * local_n + local_j) * 3;
    unsigned v = (i*n+j)*3;

    //read in the surrounding 24 pixels 

    if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
    {
        // [-2 ,-2]
        unsigned int indexLocal = ((local_i- KERNELSIZE + KERNELSIZE) * local_n + (local_j - KERNELSIZE + KERNELSIZE))*3;
        unsigned int indexGlobal = ((i - KERNELSIZE + KERNELSIZE)*n + j - KERNELSIZE + KERNELSIZE) *3;
        local_image[indexLocal + 0 ] = image[indexGlobal + 0]; 
        local_image[indexLocal + 1 ] = image[indexGlobal + 1]; 
        local_image[indexLocal + 2 ] = image[indexGlobal + 2]; 

        // [ -2, +2]
        indexLocal = ((local_i+ KERNELSIZE + KERNELSIZE) * local_n + (local_j - KERNELSIZE + KERNELSIZE))*3;
        indexGlobal = ((i + KERNELSIZE + KERNELSIZE)*n + j - KERNELSIZE + KERNELSIZE) *3;

        local_image[indexLocal + 0 ] = image[indexGlobal + 0]; 
        local_image[indexLocal + 1 ] = image[indexGlobal + 1]; 
        local_image[indexLocal + 2 ] = image[indexGlobal + 2]; 

        // [2,2]
        indexLocal = ((local_i+ KERNELSIZE + KERNELSIZE) * local_n + (local_j + KERNELSIZE + KERNELSIZE))*3;
        indexGlobal = ((i + KERNELSIZE + KERNELSIZE)*n + j + KERNELSIZE + KERNELSIZE) *3;

        local_image[indexLocal + 0 ] = image[indexGlobal + 0]; 
        local_image[indexLocal + 1 ] = image[indexGlobal + 1]; 
        local_image[indexLocal + 2 ] = image[indexGlobal + 2]; 

        // [2,-2]
        indexLocal = ((local_i- KERNELSIZE + KERNELSIZE) * local_n + (local_j + KERNELSIZE + KERNELSIZE))*3;
        indexGlobal = ((i - KERNELSIZE + KERNELSIZE)*n + j + KERNELSIZE + KERNELSIZE) *3;

        local_image[indexLocal + 0 ] = image[indexGlobal + 0]; 
        local_image[indexLocal + 1 ] = image[indexGlobal + 1]; 
        local_image[indexLocal + 2 ] = image[indexGlobal + 2];

        barrier(CLK_LOCAL_MEM_FENCE);

        /*barrier(CLK_LOCAL_MEM_FENCE);*/

        unsigned int sumx, sumy, sumz;
        int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
        //IMAGES SHOULD BE CAlocal_imageULATED IN LOCAL MEMORY AND THEN WRITTEN TO GLOBAL
        // Filter kernel
        sumx=0;sumy=0;sumz=0;
        for(int k=-KERNELSIZE;k<=KERNELSIZE;k++)
            for(int l=-KERNELSIZE;l<=KERNELSIZE;l++)	
            {
                unsigned local_index = ((local_i+KERNELSIZE+k)*(local_n)+(local_j+KERNELSIZE+l))*3;
                sumx += local_image[local_index + 0];
                sumy += local_image[local_index + 1];
                sumz += local_image[local_index + 2];
            }

        out[v+0] = sumx/divby;
        out[v+1] = sumy/divby;
        out[v+2] = sumz/divby;
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
    else
        // Edge pixels are not filtered
    {
        out[v+0] = image[v+0];
        out[v+1] = image[v+1];
        out[v+2] = image[v+2];
    }
}
/*#else //ORIGINAL*/
	/*if (j < n && i < m) // If inside image*/
	/*{*/
		/*if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)*/
		/*{*/
			/*// Filter kernel*/
			/*sumx=0;sumy=0;sumz=0;*/
			/*for(int k=-KERNELSIZE;k<=KERNELSIZE;k++)*/
				/*for(int l=-KERNELSIZE;l<=KERNELSIZE;l++)    */
				/*{*/
					/*sumx += image[((i+k)*n+(j+l))*3+0];*/
					/*sumy += image[((i+k)*n+(j+l))*3+1];*/
					/*sumz += image[((i+k)*n+(j+l))*3+2];*/
				/*}*/
			/*out[(i*n+j)*3+0] = sumx/divby;*/
			/*out[(i*n+j)*3+1] = sumy/divby;*/
			/*out[(i*n+j)*3+2] = sumz/divby;*/
		/*}*/
		/*else*/
			/*// Edge pixels are not filtered*/
		/*{*/
			/*out[(i*n+j)*3+0] = image[(i*n+j)*3+0];*/
			/*out[(i*n+j)*3+1] = image[(i*n+j)*3+1];*/
			/*out[(i*n+j)*3+2] = image[(i*n+j)*3+2];*/
		/*}*/
	/*}*/

/*#endif*/

/*}*/
