#ifndef INCLUDED_CUDA_UTILS
#define INCLUDED_CUDA_UTILS

#include <curand_kernel.h>
#include "constants.h"

//CUDA max threads per block is 1024, so a square grid of threads has
//a max length side of sqrt(1024) = 32
#define WARP_SIZE 32
#define MAX_THREAD_SQUARE_EDGE WARP_SIZE
#define MAX_THREADS 1024

//Defining here a reasonable number of threads per block to do the
//for(n_hidden) and for(n_visible) loops.
//For 2 epochs, 32 tested at best over 3 trials 10.77 seconds,
//while 16 tested at worst over 3 trials 10.67. 
#define SUB_KERNEL_NUM_THREADS 16

//This int is the number of threads per block for the parallelization
//over the inputs in a batch. It was determined with some basic
//performance testing, and 16 appeared to be the fastest. 
#define NUM_BATCH_THREADS_PER_BLOCK 16

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK_DEV(ans) { gpuAssertDevice((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

__device__ void gpuAssertDevice(cudaError_t code, const char *file, int line);


extern __constant__ int const_n_visible;
extern __constant__ int const_n_hidden;
extern __constant__ int const_batch_size;
extern __constant__ int const_k;

#ifdef WEIGHT_MATRIX_PADDING
extern __constant__ size_t const_pitch;
#ifdef MULTI_WEIGHT_MATRIX
extern __constant__ size_t const_pitch2;
#endif
#endif

__device__ double atomicAdd(double* address, double val);


int getNumBlocks(int totalThreads, int blockSize);

//PRE: matrix is in row major format with the specified width and
//     height. array has length width * height.
//POST: Writes data in matrix to array. 
/* template<typename T> */
void matrixToArray(DTYPE ** matrix, DTYPE * array, int height, int width);

void matrixToArrayTrans(DTYPE ** matrix, DTYPE * array, int height, int width);

void matrixToArrayInt(int ** matrix, int * array, int height, int width);

//PRE: matrix is in row major format with the specified width and
//     height. array has length width * height.
//POST: Writes data in array to matrix. 
/* template<typename T> */
void arrayToMatrix(DTYPE * array, DTYPE ** matrix, int height, int width);

void dims_to_num_threads_and_blocks(int x_size, int y_size,
				    dim3 & num_blocks, dim3 & num_threads);


__device__ void dims_to_num_threads_and_blocks_gpu(int x_size, int y_size,
						   dim3 & num_blocks, dim3 & num_threads);

__global__ void init_curand(curandState_t * states, int num_curand_states, int seed_add);

__device__ DTYPE * get_row_pitch_ptr(DTYPE * array, int pitch, int row_idx);

#define BLOCK_ROWS 8
  
__global__ void write_matrix_transpose(const DTYPE *W, DTYPE *W2);

#ifdef WEIGHT_MATRIX_PADDING
__global__ void write_matrix_transpose_pitch(const DTYPE *W, DTYPE *W2);
#endif

__global__
void write_bias_results_to_memory(DTYPE * data, DTYPE lr, DTYPE wc, DTYPE * ph_mean_batch,
				  DTYPE * nv_means_batch, DTYPE * nh_means_batch,
				  DTYPE * ph_sample_batch, DTYPE * nv_samples_batch,
				  DTYPE * nh_samples_batch, DTYPE * hbias,
				  DTYPE * vbias, DTYPE * dhbias, DTYPE * dvbias,
				  int data_num_rows, int data_num_cols, int curr_i);
#endif
