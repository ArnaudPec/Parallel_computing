// Reduction lab, find maximum

#include <stdio.h>
#include <math.h>
#include "milli.h"

const int tx = 512;
const int ty = 1;
const int threads_per_block = tx * ty;
const int gx = 1024*32;
const int gy = 1;
const int blocks_per_grid = gx * gy;
const int SIZE = 1024*blocks_per_grid;//2*threads_per_block;

__global__ void find_max(int *data, int size)
{
    //__shared__ int cache[threads_per_block*4];
	int tid = threadIdx.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(data[i*2] < data[i*2+1])
        data[i*2] = data[i*2+1];
    
        __syncthreads();
        data[i]=data[i*2];
    //order the data sequentially
    //for(int j = i; j < size/2; j++)
    //{
    //    data[j] = data[j*2];
    //}
}

void launch_cuda_kernel(int *data, int N)
{
	// Handle your CUDA kernel launches in this function
	
	int *devdata;
	int size = sizeof(int) * N;
	int M = N;
	int p = 0;
	while(M%2==0)
    {
        M/=2;
        p++;
    }
	int levels = 2;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );

	int o = 0;
	for(int i = 0; i < p; i++)
    {
	    //printf("threads=%i", (tx>>i));
	    int ttx = tx, ggx;
	    if(gx>>i <= 1)
        {
	        ttx = tx>>(i-o);
	        ggx = 1;
        }
        else{
	        ggx = gx>>i;
            o++;
        }
        if(ttx < 1)
	        break;
	    printf("threads=%i\n", ttx);
	    printf("blocks=%i\n", ggx);
        dim3 dimBlock( ttx, ty );
        dim3 dimGrid( ggx, gy );
        find_max<<<dimGrid, dimBlock>>>(devdata,N>>i);
        cudaThreadSynchronize();
    }    
    cudaThreadSynchronize();
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, size, cudaMemcpyDeviceToHost ); 
	/*printf("the outputed numbers\n");*/
	/*for(int i = 0; i < N; i++)*/
	/*{*/
		/*printf("%i\n", data[i]);*/
	/*}*/
	cudaFree(devdata);
}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N)
{
  int i, m;
  
	m = data[0];
	for (i=0;i<N;i++) // Loop over data
	{
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

//#define SIZE 128
//#define SIZE 16
// Dummy data in comments below for testing
int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

int main()
{
  // Generate 2 copies of random data
  srand(time(NULL));
  for (long i=0;i<SIZE;i++)
  {
    data[i] = rand() % (SIZE * 5);
    data2[i] = data[i];
  }
  
  // The GPU will not easily beat the CPU here!
  // Reduction needs optimizing or it will be slow.
  ResetMilli();
  find_max_cpu(data, SIZE);
  printf("CPU time %f\n", GetSeconds());
  ResetMilli();
  launch_cuda_kernel(data2, SIZE);
  printf("GPU time %f\n", GetSeconds());

  // Print result
  printf("\n");
  printf("CPU found max %d\n", data[0]);
  printf("GPU found max %d\n", data2[0]);
}
