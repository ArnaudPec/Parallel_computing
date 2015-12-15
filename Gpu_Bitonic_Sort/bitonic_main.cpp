
// This is not really C++-code but pretty plain C code, but we compile it
// as C++ so we can integrate with CUDA seamlessly.

#include "bitonic_kernel.h"
#include <stdio.h>
#include <math.h>
#include "milli.h"

#define SIZE 1024*128-1 //at this amount to sort the GPU becomes faster than the CPU
#define MAXPRINTSIZE 32
int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54,1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54, 1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

void generate_bitonic_list()
{
	for(int i = 0, j = SIZE; i < SIZE && j > 0; i++, j--)
	{
		//data[i] = i % 2 == 0 ? i : j;
		if(i < SIZE/2)
		{
			data[i] = i+SIZE/2+1;
			data2[i] = data[i];
		}
		else
		{
			data[i] = j;
			data2[i] = data[i];
		}
	}
}

static void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

void bitonic_cpu(int *data, int N)
{
  int i,j,k;
  for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
  {
    for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
    {
      for (i=0;i<N;i++) // Loop over data
      {
        int ixj=i^j; // Calculate indexing!
        if ((ixj)>i)
        {
          if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
          if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
        }
      }
    }
  }
}

int main()
{
	generate_bitonic_list();
  /* this is for printing the generated list
  for (int i=0;i<SIZE;i++)
    {
      printf("%d:%d\n",i, data2[i]);
    }
	*/
  ResetMilli();
  bitonic_cpu(data, SIZE);
  printf("CPU time: %f\n", GetSeconds());
  ResetMilli();
  bitonic_gpu(data2, SIZE);
  printf("GPU time: %f\n", GetSeconds());
  
  /*
  for (int i=0;i<SIZE;i++)
    {
      printf("%d:%d\n",i, data[i]);
    }
  printf("GPU:\n");
  for (int i=0;i<SIZE;i++)
    {
      printf("%d:%d\n",i, data2[i]);
    }
	*/
  for (int i=0;i<SIZE;i++)
    if (data[i] != data2[i])
    {
      printf("Error at %d! Got %d expected %d.\n", i, data2[i], data[i]);
      return(1);
    }

  // Print result
  if (SIZE <= MAXPRINTSIZE)
    for (int i=0;i<SIZE;i++)
      printf("%d ", data[i]);
  printf("\nYour sorting looks correct!\n");
}
