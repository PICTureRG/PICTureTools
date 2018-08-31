//delta product version before extracting initial cublas call. 
//Uses dynamic parallelism, where the top kernel executes over batch
//elements, and the bottom kernel executes a matrix-vector multiply. 

#include <iostream>
#include "../include/rbm_delta_product.h"
#include "../include/rbm_baseline.h"
#include "../include/utils.h"
#include "../include/cuda_utils.h"
#include "cublas_v2.h"
#include "../include/constants.h"

#include <climits> //Needed for INT_MAX

#include <curand_kernel.h>
// #include <time.h>
// #include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace utils;
#define SAMPLING_KERNEL_BLOCK_SIZE 32
//NOTE: NUM_DOT_PRODUCT_REDUCTION_THREADS must be a power of 2
// #define NUM_DOT_PRODUCT_REDUCTION_THREADS 64

#define WRITE_BIAS_KERNEL_BLOCK_SIZE 64

//NO_MORE_DIFFS is the value that will be put in the v_diffs and
//h_diffs array to indicate that we've reached the end of the
//differences found between the sampling arrays. The problem is that
//every integer could be used (of course, we probably wouldn't have
//n_visible or n_hidden be too big), so for the sake of generality, we
//set it to the max integer from <climits>
#define NO_MORE_DIFFS INT_MAX

//TODO: Switch which matrix is being accessed: this should result in a
//fair slow-down if my theories are working correctly. 
//Future work: Check which weight matrix to send to cublas.
//TODO: Check whether unified memory changed performance. 

namespace delta_product {
  //Need to declare cublas library object for device
  // __device__ cublasHandle_t dev_cublas_handle;
  
  //Call this with 1 thread to init dev_cublas_handle.
  __global__ void init_cublas_handle(cublasHandle_t * handle) {
    cublasCreate(handle);
  }
  // __global__ void init_cublas_handle_stream(cublasHandle_t * handle, cudaStream_t stream) {
  //   cublasSetStream(*handle, stream);
  // }
  
  //Call this with 1 thread to free _devcublas_handle memory. 
  __global__ void destroy_cublas_handle(cublasHandle_t * handle) {
    cublasDestroy(*handle);
  }
  
  RBM_delta::RBM_delta(int size, int n_v, int n_h, int b_size, int k,
  		       DTYPE **w, DTYPE *hb, DTYPE *vb, 
		       int data_num_rows, int data_num_cols) : baseline::RBM(size, n_v, n_h, b_size, k, w, hb, vb, data_num_rows, data_num_cols)
  {
    cudaMalloc((void**) &dev_handle, sizeof(cublasHandle_t));
    init_cublas_handle<<<1,1>>>(dev_handle);
    // if(stream != NULL) {
    //   init_cublas_handle_stream<<<1,1>>>(dev_handle, *stream);
    // }
#ifdef MULTI_WEIGHT_MATRIX
    cout << "Using MULTI_WEIGHT_MATRIX" << endl;
    if(w == NULL) {
      WArray2 = new DTYPE[n_hidden * n_visible];
    }
#endif
    cout << "RBM_delta constructor" << endl;
    cudaMalloc((void**)&dev_h_diffs, batch_size * (n_hidden + 1) * sizeof(int));
    cudaMalloc((void**)&dev_v_diffs, batch_size * (n_visible + 1) * sizeof(int));
    
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }
  
  //PRE: matrix is in row-major format with m rows and n columns. 
  //     vector has length n. result has length m. 
  //POST: writes matrix.vector to result. 
  __device__ void matrix_dot_vector(DTYPE * matrix, int m, int n, DTYPE * vector,
				    DTYPE * result, bool transpose, cublasHandle_t * handle) {
    //Note: A cublas transform is performed iff transpose is false.
    //This is because the transform is used to rotate from row major
    //to column major, as cublas needs.
    DTYPE alpha = 1.0;
    DTYPE beta = 0.0;
    #ifdef USING_DOUBLES
    cublasDgemv(*handle, transpose ? CUBLAS_OP_N : CUBLAS_OP_T,
    		n, m, &alpha, matrix, n, vector, 1, &beta, result, 1);
    #else
    cublasSgemv(*handle, transpose ? CUBLAS_OP_N : CUBLAS_OP_T,
    		n, m, &alpha, matrix, n, vector, 1, &beta, result, 1);
    #endif
  }

  //This global function is designed to do 3 things:
  //1: Basic vector addition to add bias to mean.
  //2: Set mean[i] = sigmoid(mean[i])
  //3: Run random sampling and store in sample.
  //PRE: len(mean) == len(sample) == len(bias) == length
  __global__ void finish_sampling_kernel(DTYPE * mean, DTYPE * dot_product,
					 DTYPE * sample, DTYPE * bias,
					 int length, curandState_t * curand_state_ptr) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length) {
      DTYPE mean_i = 1.0 / (1.0 + exp(-(dot_product[i] + bias[i])));
      mean[i] = mean_i;
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[i]);
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[i]);
      #endif
      sample[i] = r < mean_i;
    }
  }
  
  __device__
  void sample_h_given_v_cublas(DTYPE *v0_sample, DTYPE *mean, DTYPE * h_dot_product,
			       DTYPE *sample, DTYPE * W, DTYPE * hbias, 
			       curandState_t * curand_state_ptr, cublasHandle_t * handle) {
    matrix_dot_vector(W, const_n_hidden, const_n_visible, v0_sample, h_dot_product, false, handle);
    int num_blocks = ((const_n_hidden - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    finish_sampling_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (mean, h_dot_product, sample, hbias, const_n_hidden, curand_state_ptr);
    // cudaDeviceSynchronize();
  }

  // #ifdef DYNAMIC_DELTA_SAVE

  //BEGIN: Code from url: "https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/"
  
  #define WARP_SZ 32
  __device__
  inline int lane_id(void) {
    return threadIdx.x % WARP_SZ;
  }

  __device__ int warp_bcast(int v, int leader) {
    return __shfl(v, leader);
  }

  // warp-aggregated atomic increment
  __device__
  int atomicAggInc(int *ctr) {
    int mask = __ballot(1);
    // select the leader
    int leader = __ffs(mask) - 1;
    // leader does the update
    int res;
    if(lane_id() == leader)
      res = atomicAdd(ctr, __popc(mask));
    // broadcast result
    res = warp_bcast(res, leader);
    // each thread computes its own value
    return res + __popc(mask & ((1 << lane_id()) - 1));
  }
  //END
  
  __global__
  void save_sample_changes_kernel(DTYPE * mean, DTYPE * sample, DTYPE * prev_sample, int * diffs,
				  int length, curandState_t * curand_state_ptr) {
    // unsigned int diffs_idx_plus_1 = 0;
    // unsigned int diffs_idx = 0;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length) {
      // diffs[i] = NO_MORE_DIFFS;
      DTYPE prev_sample_i = prev_sample[i];
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[i]);
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[i]);
      #endif
      DTYPE new_sample_i = r < mean[i] ? 1.0 : 0.0;//Just to be sure it's 1 or 0
      sample[i] = new_sample_i;
      // diffs[i] = (i % 4 == 0);
      if(new_sample_i != prev_sample_i) {
      	int my_diff_value = 2 * i;
      	if(prev_sample_i != 0.0) {//Then changing to negative, so dot product decreases.
      	  my_diff_value++;
      	}
	//At the end of this function we want the value at diffs[0] to
	//represent the number of diffs that follow in the array. So
	//we can just send the diffs ptr to atomicAggInc so that it
	//modifies the first element accordingly:
      	int my_idx = atomicAggInc(diffs);
      	diffs[my_idx + 1] = my_diff_value;//+1 to avoid first index (stores size)
      }
    }
  }
  // #endif

// #ifdef DYNAMIC_DELTA_SAVE
  //Note: sample and prev_sample could be the same pointer.
  __device__ void save_sample_changes(DTYPE * mean, DTYPE * sample,
				      DTYPE * prev_sample, int * diffs,
				      int length, curandState_t * curand_state_ptr) {
    diffs[0] = 0;//Set initial diffs index counter to 0. 
    int num_blocks = ((length - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    save_sample_changes_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (mean, sample, prev_sample, diffs, length, curand_state_ptr);
    cudaDeviceSynchronize();
  }
  
  //This function is the same as finish_sampling_kernel except that it
  //doesn't do the sampling step.
  //This is useful for the first sample_v_given_h call since cublas
  //can still be used (requiring the existence of this function to
  //finish the biasing EFFICIENTLY (global mem coalescence)), but we
  //want to save changes during the sampling step. 
  //1: Basic vector addition to add bias to mean.
  //2: Set mean[i] = sigmoid(mean[i])
  //PRE: len(mean) == len(sample) == len(bias) == length
  __global__ void add_bias_kernel(DTYPE * mean, DTYPE * dot_product,
				  DTYPE * bias, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length) {
      mean[i] = 1.0 / (1.0 + exp(-(dot_product[i] + bias[i])));
    }
  }

  __device__ void add_bias(DTYPE * mean, DTYPE * dot_product,
			   DTYPE * bias, int length) {
    int num_blocks = ((length - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    add_bias_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (mean, dot_product, bias, length);
    // cudaDeviceSynchronize();
  }
  
  __device__
  void sample_v_given_h_cublas(DTYPE *h0_sample, DTYPE *v_mean, DTYPE * v_dot_product,
			       DTYPE *v_sample, DTYPE * v_prev_sample, int * v_diffs,
			       DTYPE * W, DTYPE * vbias, curandState_t * curand_state_ptr,
			       cublasHandle_t * handle) {
    // cudaDeviceSynchronize();//sync from sample_h_given_v_cublas
    matrix_dot_vector(W, const_n_hidden, const_n_visible, h0_sample,
  		      v_dot_product, true, handle);
    add_bias(v_mean, v_dot_product, vbias, const_n_visible);
    save_sample_changes(v_mean, v_sample, v_prev_sample,
			v_diffs, const_n_visible, curand_state_ptr);
    
    // int num_blocks = ((const_n_visible - 1) / DYN_CUBLAS_SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    // finish_sampling_kernel<<<num_blocks, DYN_CUBLAS_SAMPLING_KERNEL_BLOCK_SIZE>>>
    //   (mean, sample, vbias, const_n_visible, curand_state_ptr);
  }
  
  __global__
  void matrix_dot_vector_dp_kernel(DTYPE * W, int m, int n, int * diffs,
				   DTYPE * dot_product, bool transpose) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(transpose) {
      if(i < n) {
	DTYPE result_i_inc = 0.0;
	int curr_diff;
	for(int diffs_idx = 1; diffs_idx < diffs[0]+1; diffs_idx++) {
	  curr_diff = diffs[diffs_idx];
	  result_i_inc += W[(curr_diff >> 1) * n + i] * (curr_diff % 2 == 0 ? 1.0 : -1.0);
	}
	dot_product[i] += result_i_inc;
      }
    } else {
      if(i < m) {
	DTYPE result_i_inc = 0.0;
	int curr_diff;
	for(int diffs_idx = 1; diffs_idx < diffs[0]+1; diffs_idx++) {
	  curr_diff = diffs[diffs_idx];
	  #ifdef MULTI_WEIGHT_MATRIX
	  result_i_inc += W[(curr_diff >> 1) * m + i] * (curr_diff % 2 == 0 ? 1.0 : -1.0);
	  #else
	  result_i_inc += W[i * n + (curr_diff / 2)] * (curr_diff % 2 == 0 ? 1.0 : -1.0);
	  #endif
	}
	dot_product[i] += result_i_inc;
      }
    }
  }
  
  __device__ void matrix_dot_vector_dp(DTYPE * W, int m, int n, int * diffs,
				       DTYPE * dot_product, bool transpose=false) {
    //Figure out the correct length of the dot_product based on transpose:
    int len_dot_product = transpose ? n : m;
    int num_blocks = ((len_dot_product - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    matrix_dot_vector_dp_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (W, m, n, diffs, dot_product, transpose);
  }


  //PRE: prev_sample should be the last sample array that was sent
  //     into a sample_h_given_v* call.
  __device__ void sample_h_given_v_delta(DTYPE * v0_sample, DTYPE * h_mean,
					 DTYPE * h_dot_product, 
					 DTYPE * h_sample, DTYPE * prev_h_sample,
					 int * h_diffs, int * v_diffs, DTYPE * W,
					 DTYPE * hbias, curandState_t * curand_state_ptr) {
    matrix_dot_vector_dp(W, const_n_hidden, const_n_visible,
			 v_diffs, h_dot_product, false);
    add_bias(h_mean, h_dot_product, hbias, const_n_hidden);
    save_sample_changes(h_mean, h_sample, prev_h_sample, h_diffs,
			const_n_hidden, curand_state_ptr);
  }


  //PRE: prev_sample should be the last sample array that was sent
  //     into a sample_v_given_h* call.
  __device__
  void sample_v_given_h_delta(DTYPE * h0_sample, DTYPE * v_mean, DTYPE * v_dot_product, 
			      DTYPE * v_sample, DTYPE * prev_v_sample, int * h_diffs,
			      int * v_diffs, DTYPE * W, DTYPE * vbias,
			      curandState_t * curand_state_ptr) {
    // cudaDeviceSynchronize();
    // GET_TIME(t1);

    matrix_dot_vector_dp(W, const_n_hidden, const_n_visible, h_diffs,
			 v_dot_product, true);
    add_bias(v_mean, v_dot_product, vbias, const_n_visible);
    save_sample_changes(v_mean, v_sample, prev_v_sample, v_diffs, const_n_visible,
			curand_state_ptr);

    // cudaDeviceSynchronize();
    // GET_TIME(t2);
    // printf("sample_v_given_h_delta: %f\n", get_duration(t1, t2));
  }
  
  __device__
  void gibbs_hvh_delta(DTYPE *h0_sample, DTYPE *nv_means, DTYPE *nv_samples,
		       DTYPE *nh_means, DTYPE *nh_samples, DTYPE * h_dot_product,
		       DTYPE * v_dot_product, int * h_diffs, int * v_diffs, DTYPE * W,
#ifdef MULTI_WEIGHT_MATRIX
		       DTYPE * W2,
#endif
		       DTYPE * hbias, DTYPE * vbias, curandState_t * curand_state_ptr) {
    sample_v_given_h_delta(h0_sample, nv_means, v_dot_product, nv_samples, nv_samples,
			   h_diffs, v_diffs, W, vbias, curand_state_ptr);
    sample_h_given_v_delta(nv_samples, nh_means, h_dot_product, nh_samples, nh_samples,
			   h_diffs, v_diffs,
#ifdef MULTI_WEIGHT_MATRIX
			   W2,
#else
			   W,
#endif
			   hbias, curand_state_ptr);
  }
  
  __global__ void cd_gpu(DTYPE * data, int curr_i, int data_num_cols, DTYPE * W,
#ifdef MULTI_WEIGHT_MATRIX
			 DTYPE * W2,
#endif
			 DTYPE * hbias, DTYPE * vbias, curandState_t * curand_states,
			 DTYPE * ph_mean_batch, DTYPE * nv_means_batch,
			 DTYPE * nh_means_batch, DTYPE * ph_sample_batch,
			 DTYPE * nv_samples_batch, DTYPE * nh_samples_batch,
			 DTYPE * h_dot_product_batch, DTYPE * v_dot_product_batch,
			 int * h_diffs_batch, int * v_diffs_batch, int curand_batch_width,
			 cublasHandle_t * handle) {
    int batch_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(batch_i < const_batch_size) {
      DTYPE * ph_mean  = &ph_mean_batch[batch_i * const_n_hidden];
      DTYPE * nv_means = &nv_means_batch[batch_i * const_n_visible];
      DTYPE * nh_means = &nh_means_batch[batch_i * const_n_hidden];
      DTYPE * ph_sample  = &ph_sample_batch[batch_i * const_n_hidden];
      DTYPE * nv_samples = &nv_samples_batch[batch_i * const_n_visible];
      DTYPE * nh_samples = &nh_samples_batch[batch_i * const_n_hidden];

      DTYPE * h_dot_product = &h_dot_product_batch[batch_i * const_n_hidden];
      DTYPE * v_dot_product = &v_dot_product_batch[batch_i * const_n_visible];
      
      int * h_diffs = &h_diffs_batch[batch_i * (const_n_hidden + 1)];
      int * v_diffs = &v_diffs_batch[batch_i * (const_n_visible + 1)];
      // if(batch_i == 0)
      // 	printf("num_diffs: %d\n", v_diffs[0]);
      
      curandState_t * curand_state_ptr = &curand_states[batch_i * curand_batch_width];
      DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
      //The first sampling uses cublas completely.
      sample_h_given_v_cublas(input, ph_mean, h_dot_product, ph_sample,
			      W,
			      hbias, curand_state_ptr, handle);
      //The other direction uses cublas, but saves info in v_diffs
      //Note that the previous samples in this case was the input that
      //was initially provided.
      sample_v_given_h_cublas(ph_sample, nv_means, v_dot_product, nv_samples,
			      input, v_diffs, W, vbias, curand_state_ptr, handle);
      //From here out the delta product is completely used,
      //BUT, for this h->v sampling the previous h sample was
      //ph_sample (not nv_samples)
      sample_h_given_v_delta(nv_samples, nh_means, h_dot_product, nh_samples, ph_sample,
			     h_diffs, v_diffs,
#ifdef MULTI_WEIGHT_MATRIX
			     W2,//correct: W2
#else
			     W,//correct: W
#endif
			     hbias, curand_state_ptr);
      for(int step = 1; step < const_k; step++) {
	gibbs_hvh_delta(nh_samples, nv_means, nv_samples, nh_means, nh_samples,
			h_dot_product, v_dot_product,
			h_diffs, v_diffs, W,//correct: W
#ifdef MULTI_WEIGHT_MATRIX
			W2,//correct: W2
#endif
			hbias, vbias, curand_state_ptr);
      }
    }
  }


  
  __global__
  void write_weight_results_to_memory(DTYPE * data, DTYPE * W,
#ifdef MULTI_WEIGHT_MATRIX
			       DTYPE * W2,
#endif
			       DTYPE lr, DTYPE wc, DTYPE * ph_mean_batch,
			       DTYPE * nv_means_batch, DTYPE * nh_means_batch,
			       DTYPE * ph_sample_batch, DTYPE * nv_samples_batch,
			       DTYPE * nh_samples_batch, DTYPE * hbias,
			       DTYPE * vbias, DTYPE * dhbias, DTYPE * dvbias,
			       int data_num_rows, int data_num_cols, int curr_i) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if((i < const_n_hidden) && (j < const_n_visible)) {
      //Assert: dW array should be 0.
      __shared__ DTYPE dW_shared[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE];
// #ifdef MULTI_WEIGHT_MATRIX
//       __shared__ DTYPE dW2_shared[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE];
// #endif
      int shared_i = i % MAX_THREAD_SQUARE_EDGE;
      int shared_j = j % MAX_THREAD_SQUARE_EDGE;
      dW_shared[shared_i][shared_j] = 0.0;
// #ifdef MULTI_WEIGHT_MATRIX
//       dW2_shared[shared_j][shared_i] = 0.0;
// #endif
      // __shared__ DTYPE dhbias_shared[MAX_THREAD_SQUARE_EDGE];
      // __shared__ DTYPE dvbias_shared[MAX_THREAD_SQUARE_EDGE];
      //Reset the bias change counters:
      // if(j == 0) dhbias[i] = 0.0;
      // if(i == 0) dvbias[j] = 0.0;
      //Run the computations for each batch:
      for(int batch_i = 0; batch_i < const_batch_size; batch_i++) {
	DTYPE * ph_mean  = &ph_mean_batch   [batch_i * const_n_hidden];
	DTYPE * nh_means = &nh_means_batch  [batch_i * const_n_hidden];
	DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
	DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
	dW_shared[shared_i][shared_j] += ph_mean[i] * input[j] - nh_means[i] * nv_samples[j];
	// #ifdef MULTI_WEIGHT_MATRIX
	// dW2_shared[shared_j][shared_i] += dW_shared[shared_i][shared_j];
	// #endif
	// if(j == 0) dhbias[i] += ph_mean[i] - nh_means[i];
	// if(i == 0) dvbias[j] += input[j] - nv_samples[j];
      }
      //Surprisingly enough, this is not a race condition, because
      //each thread only depends on itself for this computation
      DTYPE * W_row_i = &W[const_n_visible * i];
      W_row_i[j] = W_row_i[j] + lr * (dW_shared[shared_i][shared_j] / const_batch_size - wc * W_row_i[j]);
#ifdef MULTI_WEIGHT_MATRIX
      //Just write W to W2 in transpose:
      DTYPE * W2_row_j=&W2[const_n_hidden * j];
      W2_row_j[i] = W_row_i[j];
#endif
      // if(j == 0) {
      // 	hbias[i] += lr * dhbias[i] / const_batch_size;
      // 	// dhbias[i] = 0.0;
      // }
      // if(i == 0) {
      // 	vbias[j] += lr * dvbias[j] / const_batch_size;
      // 	// dvbias[j] = 0.0;
      // }
    }
  }


  //TODO: Separate out hbias and vbias ???
    __global__
  void write_bias_results_to_memory(DTYPE * data, DTYPE * W,
#ifdef MULTI_WEIGHT_MATRIX
				    DTYPE * W2,
#endif
				    DTYPE lr, DTYPE wc, DTYPE * ph_mean_batch,
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

  void RBM_delta::reset_d_arrays() {
    //Since dW_pitch is the width of the dev_dW array rows, we
    //multiply by the number of rows (n_hidden) to get the number of
    //bytes to reset:
    // CUDA_CHECK(cudaMemset(dev_dhbias, 0, n_hidden  * sizeof(DTYPE)));
    // CUDA_CHECK(cudaMemset(dev_dvbias, 0, n_visible * sizeof(DTYPE)));
    
    //cuda_CHECK(cudaMemset(dev_ph_mean_batch   , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    //CUDA_CHECK(cudaMemset(dev_nv_means_batch  , 0, sizeof(DTYPE) * n_visible * batch_size));
    //CUDA_CHECK(cudaMemset(dev_nh_means_batch  , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    //CUDA_CHECK(cudaMemset(dev_ph_sample_batch , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    //CUDA_CHECK(cudaMemset(dev_nv_samples_batch, 0, sizeof(DTYPE) * n_visible * batch_size));
    //CUDA_CHECK(cudaMemset(dev_nh_samples_batch, 0, sizeof(DTYPE) * n_hidden  * batch_size));
  }


  void RBM_delta::contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE * dev_data) {
    reset_d_arrays();
    
    if(batch_size > MAX_THREADS) {
      cerr << "ERROR: batch_size cannot exceed 1024" << endl;
    }
    
    // GET_TIME(k1_t1);
    // cerr << "time: " << k1_t1 << endl;
    int cd_blocks = 1 + (batch_size - 1) / NUM_BATCH_THREADS_PER_BLOCK;
    // #ifdef SIMULTANEOUS_EXECUTION
    cd_gpu <<< cd_blocks, NUM_BATCH_THREADS_PER_BLOCK, 0, stream>>>
      (dev_data, curr_i, data_num_cols, dev_W,
       #ifdef MULTI_WEIGHT_MATRIX
       dev_W2,
       #endif
       dev_hbias, dev_vbias,
       dev_curand_states, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_h_dot_product_batch, dev_v_dot_product_batch,
       dev_h_diffs, dev_v_diffs, curand_batch_width, dev_handle);
    // #else
    // cd_gpu <<< cd_blocks, NUM_BATCH_THREADS_PER_BLOCK>>>
    //   (dev_data, curr_i, data_num_cols, dev_W, dev_hbias, dev_vbias,
    //    dev_curand_states, dev_ph_mean_batch, dev_nv_means_batch,
    //    dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
    //    dev_nh_samples_batch, dev_h_dot_product_batch, dev_v_dot_product_batch,
    //    dev_h_diffs, dev_v_diffs, curand_batch_width, dev_handle);
    // #endif
    //temp:
    // GET_TIME(k1_t2);
    // cerr << "k1 time: " << get_duration(k1_t1, k1_t2) << endl;
    
    dim3 num_blocks, num_threads;
    dims_to_num_threads_and_blocks(n_visible, n_hidden, num_blocks, num_threads);
    // GET_TIME(k2_t1);
    
    // #ifdef SIMULTANEOUS_EXECUTION
    write_weight_results_to_memory <<< num_blocks, num_threads, 0, stream>>>
      (dev_data, dev_W,
       #ifdef MULTI_WEIGHT_MATRIX
       dev_W2,
       #endif
       lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);
    int most_nodes = max(n_hidden, n_visible);
    int num_bias_blocks = 1 + ((most_nodes-1) / WRITE_BIAS_KERNEL_BLOCK_SIZE);
    int num_bias_threads = WRITE_BIAS_KERNEL_BLOCK_SIZE;
    write_bias_results_to_memory <<< num_bias_blocks, num_bias_threads, 0, stream>>>
      (dev_data, dev_W,
       #ifdef MULTI_WEIGHT_MATRIX
       dev_W2,
       #endif
       lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);    // #else
    // write_results_to_memory <<< num_blocks, num_threads>>>
    //   (dev_data, dev_W, lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
    //    dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
    //    dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
    //    data_num_rows, data_num_cols, curr_i);
    // #endif
    //temp:
    // GET_TIME(k2_t2);
    // cerr << "k2 time: " << get_duration(k2_t1, k2_t2) << endl;
  }

  
  void RBM_delta::allocate_special_memory() {
    // data = new DTYPE[data_num_rows * data_num_cols];
    // for(int i = 0; i < data_num_rows * data_num_cols; i++) {
    //   data[i] = (DTYPE) int_data[i];
    // }
    cerr << "allocate special\n";
    CUDA_CHECK(cudaMalloc((void**)&dev_ph_sample_batch , sizeof(DTYPE) * n_hidden  * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nv_samples_batch, sizeof(DTYPE) * n_visible * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nh_samples_batch, sizeof(DTYPE) * n_hidden  * batch_size));
    
    CUDA_CHECK(cudaMalloc((void**)&dev_W , n_hidden * n_visible * sizeof(DTYPE)));
    matrixToArray     (W, WArray , n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(dev_W, WArray, n_hidden * n_visible * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    
    #ifdef MULTI_WEIGHT_MATRIX
    CUDA_CHECK(cudaMalloc((void**)&dev_W2, n_hidden * n_visible * sizeof(DTYPE)));
    matrixToArrayTrans(W, WArray2, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(dev_W2, WArray2, n_hidden * n_visible * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    #endif
    
    // CUDA_CHECK(cudaMalloc((void**)&dev_data,
    // 			  data_num_rows * data_num_cols * sizeof(DTYPE)));
    // CUDA_CHECK(cudaMemcpy(dev_data, data, data_num_rows * data_num_cols * sizeof(DTYPE),
    // 			  cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&dev_h_dot_product_batch,
			  sizeof(DTYPE) * n_hidden * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_v_dot_product_batch,
			  sizeof(DTYPE) * n_visible  * batch_size));
    cerr << "allocate special DONE\n";
  }

  void RBM_delta::copy_matrices_to_host() {
    CUDA_CHECK(cudaMemcpy(WArray, dev_W, n_hidden * n_visible * sizeof(DTYPE), 
    			  cudaMemcpyDeviceToHost));
    arrayToMatrix(WArray, W, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(vbias, dev_vbias, n_visible * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbias, dev_hbias, n_hidden  * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
  }

  // void RBM_delta::saveWeightMatrix() {
  //   cout << "delta_product: Saving weight matrix" << endl;
  //   copy_matrices_to_host();
  //   matrixToArray (W, WArray, n_hidden, n_visible);
  //   string wFilename(MATRIX_FILENAME);
  //   saveMatrix(WArray, (size_t) n_hidden, (size_t) n_visible, wFilename);
  //   string hbiasFilename("hbias.dat");
  //   saveArray(hbias, (size_t) n_hidden, hbiasFilename);
  //   string vbiasFilename("vbias.dat");
  //   saveArray(vbias, (size_t) n_visible, vbiasFilename);
  // }


  RBM_delta::~RBM_delta() {
    // saveWeightMatrix();
    cudaFree(dev_h_diffs);
    cudaFree(dev_v_diffs);
    cudaFree(dev_h_dot_product_batch);
    cudaFree(dev_v_dot_product_batch);
    destroy_cublas_handle<<<1,1>>>(dev_handle);
    cudaFree(dev_handle);
    
    #ifdef MULTI_WEIGHT_MATRIX
    delete[] WArray2;
    cudaFree(dev_W2);
    #endif
  }
}
