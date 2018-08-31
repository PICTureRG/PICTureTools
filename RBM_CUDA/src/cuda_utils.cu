#include "../include/cuda_utils.h"
#include <cstdio>

__constant__ int const_n_visible;
__constant__ int const_n_hidden;
__constant__ int const_batch_size;
__constant__ int const_k;

#ifdef WEIGHT_MATRIX_PADDING
__constant__ size_t const_pitch;
#ifdef MULTI_WEIGHT_MATRIX
__constant__ size_t const_pitch2;
#endif
#endif

/* Don't need when using cuda 8.0 apparently
//From CUDA programming guide:
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
*/

int getNumBlocks(int totalThreads, int blockSize) {
  return(((totalThreads-1)/blockSize) + 1);
}

void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

__device__ void gpuAssertDevice(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("Dynamic CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}


// template<typename T>
void matrixToArray(DTYPE ** matrix, DTYPE * array, int height, int width) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      array[row * width + col] = matrix[row][col];
    }
  }
}

//For use with MULTI_WEIGHT_MATRIX active
void matrixToArrayTrans(DTYPE ** matrix, DTYPE * array, int height, int width) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      array[col * height + row] = matrix[row][col];
    }
  }
}

void matrixToArrayInt(int ** matrix, int * array, int height, int width) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      array[row * width + col] = matrix[row][col];
    }
  }
}

// template<typename T>
void arrayToMatrix(DTYPE * array, DTYPE ** matrix, int height, int width) {
  for(int i = 0; i < width * height; i++) {
    matrix[i / width][i % width] = array[i];
  }
}

void dims_to_num_threads_and_blocks(int x_size, int y_size,
				    dim3 & num_blocks, dim3 & num_threads) {
  num_blocks.x = (x_size / MAX_THREAD_SQUARE_EDGE) + 1;//TODO: Fix
  num_blocks.y = (y_size / MAX_THREAD_SQUARE_EDGE) + 1;
  num_threads.x = MAX_THREAD_SQUARE_EDGE;
  num_threads.y = MAX_THREAD_SQUARE_EDGE;
}

__device__ void dims_to_num_threads_and_blocks_gpu(int x_size, int y_size,
						   dim3 & num_blocks, dim3 & num_threads) {
  num_blocks.x = (x_size / MAX_THREAD_SQUARE_EDGE) + 1;
  num_blocks.y = (y_size / MAX_THREAD_SQUARE_EDGE) + 1;
  num_threads.x = MAX_THREAD_SQUARE_EDGE;
  num_threads.y = MAX_THREAD_SQUARE_EDGE;
}


//PRE: states is allocated with length num_curand_states.
//     batch_size < 1024, batch_size threads are being executed.
__global__ void init_curand(curandState_t * states, int num_curand_states, int seed_add) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < num_curand_states) {
    curand_init(seed_add + i, i, 0, &states[i]);
  }
}

//Takes an array allocated with cudaMallocPitch, the pitch returned
//in cudaMallocPitch, and the row idx.
//Returns a ptr to the specified row.
__device__ DTYPE * get_row_pitch_ptr(DTYPE * array, int pitch, int row_idx) {
  return((DTYPE*) (((char*) array) + row_idx * pitch));
}


__global__
void write_bias_results_to_memory(DTYPE * data, DTYPE lr, DTYPE wc, DTYPE * ph_mean_batch,
				  DTYPE * nv_means_batch, DTYPE * nh_means_batch,
				  DTYPE * ph_sample_batch, DTYPE * nv_samples_batch,
				  DTYPE * nh_samples_batch, DTYPE * hbias,
				  DTYPE * vbias, DTYPE * dhbias, DTYPE * dvbias,
				  int data_num_rows, int data_num_cols, int curr_i) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if((i < const_n_hidden) || (i < const_n_visible)) {//i < max(const_n_hidden, const_n_visible)
    //Reset the bias change counters:
    if(i < const_n_hidden ) dhbias[i] = 0.0;
    if(i < const_n_visible) dvbias[i] = 0.0;
    //Run the computations for each batch:
    for(int batch_i = 0; batch_i < const_batch_size; batch_i++) {
      DTYPE * ph_mean  = &ph_mean_batch   [batch_i * const_n_hidden];
      DTYPE * nh_means = &nh_means_batch  [batch_i * const_n_hidden];
      DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
      DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
      if(i < const_n_hidden ) dhbias[i] += ph_mean[i] - nh_means[i];
      if(i < const_n_visible) dvbias[i] += input[i] - nv_samples[i];
    }
      
    if(i < const_n_hidden ) hbias[i] += lr * dhbias[i] / const_batch_size;
    if(i < const_n_visible) vbias[i] += lr * dvbias[i] / const_batch_size;
  }
}


__global__ void write_matrix_transpose(const DTYPE *W, DTYPE *W2)
{
  __shared__ float tile[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE+1];
    
  int x = blockIdx.x * MAX_THREAD_SQUARE_EDGE + threadIdx.x;
  int y = blockIdx.y * MAX_THREAD_SQUARE_EDGE + threadIdx.y;
  // int width = gridDim.x * MAX_THREAD_SQUARE_EDGE;
  if(x < const_n_visible) {
    for (int j = 0; (j < MAX_THREAD_SQUARE_EDGE) && (y+j < const_n_hidden); j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = W[(y+j)*const_n_visible + x];
  }
    
  __syncthreads();

  x = blockIdx.y * MAX_THREAD_SQUARE_EDGE + threadIdx.x;  // transpose block offset
  y = blockIdx.x * MAX_THREAD_SQUARE_EDGE + threadIdx.y;

  if(x < const_n_hidden) {
    for (int j = 0; (j < MAX_THREAD_SQUARE_EDGE) && (y+j < const_n_visible); j += BLOCK_ROWS)
      // for (int j = 0; j < MAX_THREAD_SQUARE_EDGE; j += BLOCK_ROWS)
      W2[(y+j)*const_n_hidden + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

#ifdef WEIGHT_MATRIX_PADDING
__global__ void write_matrix_transpose_pitch(const DTYPE *W, DTYPE *W2)
{
  __shared__ float tile[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE+1];
    
  int x = blockIdx.x * MAX_THREAD_SQUARE_EDGE + threadIdx.x;
  int y = blockIdx.y * MAX_THREAD_SQUARE_EDGE + threadIdx.y;
  // int width = gridDim.x * MAX_THREAD_SQUARE_EDGE;
  if(x < const_n_visible) {
    for (int j = 0; (j < MAX_THREAD_SQUARE_EDGE) && (y+j < const_n_hidden); j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = W[(y+j)*(const_pitch/sizeof(DTYPE)) + x];
  }
  
  __syncthreads();

  x = blockIdx.y * MAX_THREAD_SQUARE_EDGE + threadIdx.x;  // transpose block offset
  y = blockIdx.x * MAX_THREAD_SQUARE_EDGE + threadIdx.y;

  if(x < const_n_hidden) {
    for (int j = 0; (j < MAX_THREAD_SQUARE_EDGE) && (y+j < const_n_visible); j += BLOCK_ROWS)
      // for (int j = 0; j < MAX_THREAD_SQUARE_EDGE; j += BLOCK_ROWS)
      W2[(y+j)*(const_pitch2/sizeof(DTYPE)) + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}
#endif
