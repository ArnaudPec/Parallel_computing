#include <cstdio>
#include <algorithm>
#include <pthread.h>
#include <string.h>

#include "sort.h"

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compareile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
int *begin;
#define debug(var) printf("[%s:%s:%d] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#else
#define show(first, last)
#define show_ptr(first, last)
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif


// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray
{
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	public:
	DynArray(T* buffer, size_t size)
	{
		this->buffer = buffer;
		this->size = size;
	}

	iterator begin()
	{
		return buffer;
	}

	iterator end()
	{
		return buffer + size;
	}

	protected:
		T* buffer;
		size_t size;
};

static
void
cxx_sort(int *array, size_t size)
{
	DynArray<int> cppArray(array, size);
	std::sort(cppArray.begin(), cppArray.end());
}

/*
static void simple_qs(struct chunck *chuncks){
    int *array = chuncks->array;
    size_t size = chuncks->size;

	simple_quicksort(array, size);

}*/
// A very simple quicksort implementation
// * Recursion until array size is 1
// * Bad pivot picking
// * Not in place
static
void
simple_quicksort(int *array, size_t size)
{
	int pivot, pivot_count, i;
	int *left, *right;
	size_t left_size = 0, right_size = 0;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(size > 1)
	{
		// Bad, bad way to pick a pivot
		// Better take a sample and pick
		// it median value.
		pivot = array[size / 2];
		
		left = (int*)malloc(size * sizeof(int));
		right = (int*)malloc(size * sizeof(int));

		// Split
		for(i = 0; i < size; i++)
		{
			if(array[i] < pivot)
			{
				left[left_size] = array[i];
				left_size++;
			}
			else if(array[i] > pivot)
			{
				right[right_size] = array[i];
				right_size++;
			}
			else
			{
				pivot_count++;
			}
		}

		// Recurse		
		simple_quicksort(left, left_size);
		simple_quicksort(right, right_size);

		// Merge
		memcpy(array, left, left_size * sizeof(int));
		for(i = left_size; i < left_size + pivot_count; i++)
		{
			array[i] = pivot;
		}
		memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

		// Free
		free(left);
		free(right);
	}
	else
	{
		// Do nothing
	}
}

// This is used as sequential sort in the pipelined sort implementation with drake (see merge.c)
// to sort initial input data chunks before streaming merge operations.
void
sort(int* array, size_t size)
{
	// Do some sorting magic here. Just remember: if NB_THREADS == 0, then everything must be sequential
	// When this function returns, all data in array must be sorted from index 0 to size and not element
	// should be lost or duplicated.

	// Use preprocessor directives to influence the behavior of your implementation. For example NB_THREADS denotes
	// the number of threads to use and is defined at compareile time. NB_THREADS == 0 denotes a sequential version.
	// NB_THREADS == 1 is a parallel version using only one thread that can be useful to monitor the overhead
	// brought by addictional parallelization code.

	// This is to make the base skeleton to work. Replace it with your own implementation
	simple_quicksort(array, size);
	// Use C++ sequential sort, just to see how fast it is
	//cxx_sort(array, size);

	// Reproduce this structure here and there in your code to compare sequential or parallel versions of your code.
#if NB_THREADS == 0
	// Some sequential-specific sorting code
	simple_quicksort(array, size);
#else
	// Some parallel sorting-related code
	init_threads(array, size);
#endif // #if NB_THREADS
}

#if NB_THREADS > 0

void init_threads(int* c_array, int size)
{
    vector<int> array;
    for(int i = 0; i < size; i++)
    {
        array.push_back(c_array[i]);
    }

	int i,j;

     //struct chunck chuncks[NB_THREADS];// = (chunck*)malloc(sizeof(chunck));
    vector<chunck> chuncks(NB_THREADS);
    //chuncks->array = (int*)malloc(sizeof(size/NB_THREADS));
    //chuncks->sorted_array;// = malloc(sizeof(chuncks->array));
    //chuncks->pivots = malloc(sizeof(int)*NB_THREADS-1);
    pthread_t thread[NB_THREADS];
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    for(j=0; j < NB_THREADS; j++)
    {
        for(i=0; i < NB_THREADS-1; i++)
        {
            //could be picked better maybe
            chuncks[j].pivots[i] = array[(size/NB_THREADS)*i];
        }
        simple_quicksort(chuncks[i].pivots, NB_THREADS-1); 
    }
    

    for(i=0; i < NB_THREADS-1; i++)
    {
        //could be picked better maybe
        //chuncks[i].array();
        printf("%s %i: %i\n","pivot ", i, chuncks[i].pivots[i]);
    }

    /*
    for(int i = 0; i < array.size(); ++i)
    
    {
        cout << "original value " << array[i] << endl;
    }
    */

   for(i=0; i<NB_THREADS; i++){
       int jump = i*size/NB_THREADS;
       int jump2 = (i+1)*size/NB_THREADS > array.size() ? (i+1)*size/NB_THREADS  : array.size()-1;
       //chuncks[i].array(array.begin()+jump, array.begin()+jump2);
       //cout << chuncks[i].array.empty() << endl;
       copy(array.begin()+jump, array.begin()+jump2, back_inserter(chuncks[i].array));

        chuncks[i].id = i;
        printf("%s : %i\n","creating threads", chuncks[i].id);
   }

   /*
    for(int i = 0; i < chuncks[0].array.size(); ++i)
    {
        cout << "copied value " << chuncks[0].array[i] << endl;
    }
    */

   for(i=0; i<NB_THREADS; i++){
        pthread_create(&thread[i], 
                &thread_attr, 
                &qs, 
                &chuncks[i]);
    }
   for(i=0; i<NB_THREADS; i++){
        pthread_join(thread[i], NULL);
        //inital sorting done
    }
   
   //merge what has been sorted so far.
   vector<vector<int > > merge_sorted (4);
   // for(int i = 0;i < NB_THREADS; i++)
   // {
   //     merge_sorted.push_back(*new vector<int>());
   // }
   for(int i = 0; i < chuncks.size(); i++)
   {
       for(int j = 0; j < chuncks[i].sorted_arrays.size(); j++ )
       {
            copy(chuncks[i].sorted_arrays[j].begin(), chuncks[i].sorted_arrays[j].end(), back_inserter(merge_sorted[j]));
           //merge_sorted.push_back(chuncks[i].sorted_arrays[j]) 
       }
   }

    for(int j=0; j< merge_sorted.size(); j++){
        if(j < NB_THREADS)
            cout << "Following values should be less than pivot: " << chuncks[0].pivots[j] << endl;
        //else
            //cout << "largest values:" << endl;
        cout << "list size: " << merge_sorted[j].size() << endl;
        for(int i= 0; i < merge_sorted[j].size(); i++)
        {
            cout << merge_sorted[j][i] << endl;
        }
    }
   //divide up the work again among the threads
  /* 
   for(i=0; i<NB_THREADS; i++){
       move(merge_sorted[i].begin(), merge_sorted[i].end(), chunck[i].array.begin());
        pthread_create(&thread[i], 
                &thread_attr, 
                &q, 
                &chuncks[i]);
    }
   for(i=0; i<NB_THREADS; i++){
        pthread_join(thread[i], NULL);
        //inital sorting done
    }
    */
}


static void* qs(void* input)
{
    struct chunck* data;
    data = (struct chunck*) input;
    for(int i = 0;i < NB_THREADS; i++)
    {
        data->sorted_arrays.push_back(*new vector<int>());
    }

    printf("thread id: %i\n", data->id);
    //sort according to pivots
    for(int i = 0; i < data->array.size(); ++i)
    {
        //cout << "sort value " << data->array[i] << endl;
        //printf("%i : %s\n",data->id, "start sorting");
        bool nothing_pushed = true;
        for(int j = 0; j < NB_THREADS-1; j++)
        {
             //printf("%i : %s\n",data->id, "inner loop");
            if(data->array[i] < data->pivots[j])
            {
                data->sorted_arrays[j].push_back(data->array[i]);
                data->array.erase(data->array.begin()+i);
                nothing_pushed = false;
            }
            else if(j == NB_THREADS-2)
            {
                data->sorted_arrays[j+1].push_back(data->array[i]);
                nothing_pushed = false;
            }
        }
            if(nothing_pushed)
                cout << "nothing was pushed!" << endl;
    }
    
    

}
#endif
