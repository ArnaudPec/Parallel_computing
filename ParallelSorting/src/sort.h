#ifdef __cplusplus
#include <vector>
#include <iostream>
#include <iterator>
using namespace std;
#if NB_THREADS > 0

struct chunck{
    int id;
    vector<int> array;
    int pivots[NB_THREADS-1];
    vector<vector<int> > sorted_arrays;
};

static void* qs(void* data);

void init_threads(int*array, int size);

#endif


extern "C" {
#endif

#include <stddef.h>
// Some sort implementation
void sort(int* array, size_t size);

#ifdef __cplusplus
}
#endif
