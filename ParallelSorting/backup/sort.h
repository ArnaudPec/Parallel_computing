#ifdef __cplusplus
#include <vector>
#include <iostream>
#include <iterator>
#include <utility>
using namespace std;
#if NB_THREADS > 0

struct chunck{
    int id;
    vector<int> array;
    int pivots[NB_THREADS-1];
    vector<vector<int> > sorted_arrays; //partial sortings that is done in qs
	vector<int> actual_sortings; //finished sorting that is done in q
};

struct c_array
{
	int* array;
	size_t size;
};

static void* qs(void* data); //partially sort the data according to some pivots
static void* q(void* data); //sorts what qs has partially sorted.
c_array* vector_to_c_array(vector<int>& v);


void init_threads(int*array, int size);

#endif


extern "C" {
#endif

static void* simple_qsort_wrapper(void* data);
#include <stddef.h>
// Some sort implementation
void sort(int* array, size_t size);

#ifdef __cplusplus
}
#endif
