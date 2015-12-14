/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2
#define WORK_GROUP_DIM 16
#define LOCAL_INNER_SIZE 16*16*3
#define LOCAL_OUTER_SIZE 18*18*3
#define LOCAL true

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{ 
	unsigned int i = get_global_id(1) % 512;
	unsigned int j = get_global_id(0) % 512;

	//size_t wg_i = get_group_id(1) % WORK_GROUP_DIM*2;
	//size_t wg_j = get_group_id(0) % WORK_GROUP_DIM*2;

	//size of local image
	unsigned local_n = n/32;
	unsigned local_m = m/32;
	//unsigned local_m = m/32;
	unsigned local_i = get_local_id(1);//i + wg_i * WORK_GROUP_DIM;
	unsigned local_j = get_local_id(0);//i + wg_i * WORK_GROUP_DIM;

	__local unsigned char local_image[LOCAL_OUTER_SIZE]; // will contain part of image to be blurred and necessary surrounding pixels
	__local unsigned char local_out[LOCAL_INNER_SIZE]; // will contain the blurred imaged without the surrounding pixels
	unsigned local_v = (local_i * local_n + local_j) * 3;
	unsigned local_inner = (local_i * local_n + local_j) * 3;
	unsigned v = (i*n+j)*3;

	//just a test of copying image to out;
	/*
	   out[(i*n+j)*3+0] = image[v+0];
	   out[(i*n+j)*3+1] = image[v+1];
	   out[(i*n+j)*3+2] = image[v+2];
	 */
	//read part of image into local memory. 
	//read the pixel to be moddified into local memory
	//local_image[local_v+0] = image[v+0];
	//local_image[local_v+1] = image[v+1];
	//local_image[local_v+2] = image[v+2];
	//local_image[local_inner+0] = image[v+0];
	//local_image[local_inner+1] = image[v+1];
	//local_image[local_inner+2] = image[v+2];

	int k, l;
	//read in the surrounding 24 pixels 

	if (local_i >= KERNELSIZE && local_i < local_m-KERNELSIZE && local_j >= KERNELSIZE && j < n-KERNELSIZE)
		if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE )
	//if (local_i+2 >= KERNELSIZE && local_i+2 < local_m-KERNELSIZE && local_j+2 >= KERNELSIZE && j+2 < n-KERNELSIZE)
	//	if (i+2 >= KERNELSIZE && i+2 < m-KERNELSIZE && j+2 >= KERNELSIZE && j+2 < n-KERNELSIZE )
		{
			//access all the surrounding elements
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++)	
				{
					unsigned index = ((i+k)*n+(j+l))*3;
					unsigned local_index = ((local_i+k)*(local_n)+(local_j+l))*3;
					local_image[local_index+0] = image[index + 0];
					local_image[local_index+1] = image[index + 1];
					local_image[local_index+2] = image[index + 2];
				}
		}

	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int sumx, sumy, sumz;
	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
#if LOCAL //that is run algorithm using local memory
	//IMAGES SHOULD BE CALCULATED IN LOCAL MEMORY AND THEN WRITTEN TO GLOBAL
	if (j < n && i < m) // If inside image
	{

		if (local_i >= KERNELSIZE && local_i < local_m-KERNELSIZE && local_j >= KERNELSIZE && j < n-KERNELSIZE)
			//if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
			{
				// Filter kernel
				sumx=0;sumy=0;sumz=0;
				for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
					for(l=-KERNELSIZE;l<=KERNELSIZE;l++)	
					{
						//unsigned index = ((i+k)*n+(j+l))*3;
						unsigned local_index = ((local_i+k)*(local_n)+(local_j+l))*3;
						sumx += local_image[local_index + 0];
						sumy += local_image[local_index + 1];
						sumz += local_image[local_index + 2];
					}
				local_out[local_v+0] = sumx/divby;
				local_out[local_v+1] = sumy/divby;
				local_out[local_v+2] = sumz/divby;
				//barrier(CLK_LOCAL_MEM_FENCE);
			}
			else
				// Edge pixels are not filtered
			{
				local_out[local_v+0] = 255;local_image[local_v+0];
				local_out[local_v+1] = 255;local_image[local_v+1];
				local_out[local_v+2] = 255;local_image[local_v+2];
			}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	out[(i*n+j)*3+0] = local_out[local_v+0];
	out[(i*n+j)*3+1] = local_out[local_v+1];
	out[(i*n+j)*3+2] = local_out[local_v+2];
	//out[(i*n+j)*3+0] = local_image[local_inner+0];
	//out[(i*n+j)*3+1] = local_image[local_inner+1];
	//out[(i*n+j)*3+2] = local_image[local_inner+2];
	//out[(i*n+j)*3+0] = local_image[local_v+0];
	//out[(i*n+j)*3+1] = local_image[local_v+1];
	//out[(i*n+j)*3+2] = local_image[local_v+2];
	//barrier(CLK_LOCAL_MEM_FENCE);

#else
	if (j < n && i < m) // If inside image
	{
		if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
		{
			// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++)    
				{
					sumx += image[((i+k)*n+(j+l))*3+0];
					sumy += image[((i+k)*n+(j+l))*3+1];
					sumz += image[((i+k)*n+(j+l))*3+2];
				}
			out[(i*n+j)*3+0] = sumx/divby;
			out[(i*n+j)*3+1] = sumy/divby;
			out[(i*n+j)*3+2] = sumz/divby;
		}
		else
			// Edge pixels are not filtered
		{
			out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
			out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
			out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
		}
	}

#endif

}
