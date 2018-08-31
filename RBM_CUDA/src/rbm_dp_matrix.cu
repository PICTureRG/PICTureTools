#include <iostream>
#include "../include/rbm_dp_matrix.h"
#include "../include/rbm_baseline.h"
#include "../include/utils.h"
#include "../include/cuda_utils.h"
#include "../include/write_weights_kernel.h"
#include "cublas_v2.h"
#include "../include/constants.h"
#include <stdio.h>

#include <climits> //Needed for INT_MAX

#include <curand_kernel.h>

#include <vector>

using namespace std;
using namespace utils;

//with weight matrix padding and matrix transpose, things are complicated
//pitch and pitch2 are the pitches of dev_W and dev_W2.
//dev_W_cublas is the non-pitched memory for cublas to use.
//write_matrix_transpose_pitch is the transpose writer kernel for pitch matrices. 

// #define DEBUG

#define DPMM_Y_BLOCK_SIZE 16

#define SAMPLING_KERNEL_BLOCK_SIZE 1024
#define ADD_BIAS_KERNEL_BLOCK_SIZE 1024

#define WRITE_BIAS_KERNEL_BLOCK_SIZE 64

//Future work: Check which weight matrix to send to cublas.
namespace dp_matrix {
  //Need to declare cublas library object for device
  // __device__ cublasHandle_t dev_cublas_handle;


  //Pre: All arrays/matrices are in row-major format.
  //     A, B, C must be CUDA device pointers.
  //     All other parameters are host data (including the handle). 
  //     A and B should contain data.
  //     The data in C has no effect on the program result. 
  //      A is m rows and k columns
  //      B is k rows and n columns
  //      C is m rows and n columns
  //     These dimensions are the dimensions AFTER the specified
  //     transposition has been done. They are essentially the 3
  //     dimensions specifying the multiplication that will actually
  //     happen. 
  //     transA is whether matrix A should be transposed.
  //     transB is whether matrix B should be transposed.
  //     handle is the initialized cublas handle.
  //Post: Sets C to the result of the matrix operation A.B, where A and
  //      B have been transposed as specified by transA and transB. 
  void gemm(DTYPE * A, DTYPE * B, DTYPE * C,
	    int m, int n, int k, bool transA, bool transB, cublasHandle_t & handle) {
    //This function has a lot of nit-picky complexity due to the column
    //major formatting required by cublas input.
    //Basically, A and B are being switched in order to get the
    //transposed output of C in row major format. This requires an extra
    //transpose, which cancels out the transpose that was needed to
    //convert from row major to column major. The leading dimensions
    //also conditionally change.... 
    DTYPE alpha = 1.0;
    DTYPE beta = 0.0;
    int lda = transA ? m : k;
    int ldb = transB ? k : n;
    int ldc = n;//lol not mistake
    cublasStatus_t status;

    // cudaDeviceSynchronize();
    // double t1 = get_wall_time();
    //Need B.A in order to get transposed result in C.
    #ifdef USING_DOUBLES

    // cublas_gemm: 0.000730038
    // cublas_gemm: 0.000989199
    status = cublasDgemm(handle,
			 transB ? CUBLAS_OP_T : CUBLAS_OP_N,
			 transA ? CUBLAS_OP_T : CUBLAS_OP_N,
			 n, m, k,
			 &alpha,
			 B, ldb,
			 A, lda,
			 &beta,
			 C, ldc);
    #else
    // cublas_gemm: 0.000174046
    // cublas_gemm: 0.000134945
    status = cublasSgemm(handle,
			 transB ? CUBLAS_OP_T : CUBLAS_OP_N,
			 transA ? CUBLAS_OP_T : CUBLAS_OP_N,
			 n, m, k,
			 &alpha,
			 B, ldb,
			 A, lda,
			 &beta,
			 C, ldc);
    #endif
    // cudaDeviceSynchronize();
    // double t2 = get_wall_time();
    // cout << "cublas_gemm: " << (t2-t1) << endl;
    
    if(status != CUBLAS_STATUS_SUCCESS) {
      cerr << "gemm error\n";
      if(status == CUBLAS_STATUS_NOT_INITIALIZED) {
	cerr << "CUBLAS_STATUS_NOT_INITIALIZED\n";
      } else if(status == CUBLAS_STATUS_ALLOC_FAILED) {
	cerr << "CUBLAS_STATUS_ALLOC_FAILED\n";
      } else if(status == CUBLAS_STATUS_INVALID_VALUE) {
	cerr << "CUBLAS_STATUS_INVALID_VALUE\n";
      } else if(status == CUBLAS_STATUS_ARCH_MISMATCH) {
	cerr << "CUBLAS_STATUS_ARCH_MISMATCH\n";
      } else if(status == CUBLAS_STATUS_MAPPING_ERROR) {
	cerr << "CUBLAS_STATUS_MAPPING_ERROR\n";
      } else if(status == CUBLAS_STATUS_EXECUTION_FAILED) {
	cerr << "CUBLAS_STATUS_EXECUTION_FAILED\n";
      } else if(status == CUBLAS_STATUS_INTERNAL_ERROR) {
	cerr << "CUBLAS_STATUS_INTERNAL_ERROR\n";
      } else if(status == CUBLAS_STATUS_NOT_SUPPORTED) {
	cerr << "CUBLAS_STATUS_NOT_SUPPORTED\n";
      } else if(status == CUBLAS_STATUS_LICENSE_ERROR) {
	cerr << "CUBLAS_STATUS_LICENSE_ERROR\n";
      } else {
	cerr << "Unknown CUBLAS error\n";
      }
    }
  }

  
  
  RBM_dp_matrix::RBM_dp_matrix(int size, int n_v, int n_h, int b_size, int k,
  		       DTYPE **w, DTYPE *hb, DTYPE *vb, 
		       int data_num_rows, int data_num_cols) : baseline::RBM(size, n_v, n_h, b_size, k, w, hb, vb, data_num_rows, data_num_cols)
  {
    // cudaMalloc((void**) &dev_handle, sizeof(cublasHandle_t));
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasCreate(&host_handle);
    // init_cublas_handle<<<1,1>>>(dev_handle);
    // if(stream != NULL) {
    //   init_cublas_handle_stream<<<1,1>>>(dev_handle, *stream);
    // }
#ifdef MULTI_WEIGHT_MATRIX
    cout << "Using MULTI_WEIGHT_MATRIX" << endl;
    if(w == NULL) {
      WArray2 = new DTYPE[n_hidden * n_visible];
    }
#endif

    #ifndef BIT_CODING
    CUDA_CHECK(cudaMalloc((void**)&hdirections, batch_size * n_hidden  * sizeof(bool)));
    CUDA_CHECK(cudaMalloc((void**)&vdirections, batch_size * n_visible * sizeof(bool)));
    #endif
    
    cout << "RBM_dp_matrix constructor" << endl;
    CUDA_CHECK(cudaMalloc((void**)&dev_h_diffs, batch_size * (n_hidden + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_v_diffs, batch_size * (n_visible + 1) * sizeof(int)));
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  }



  //=========================================================
  //Primary RBM calculation methods and kernels
  //=========================================================


  __global__ void finish_sampling_kernel_matrix(DTYPE * mean, DTYPE * dot_product,
						DTYPE * sample, DTYPE * bias,
						int length, curandState_t * curand_states) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length * const_batch_size) {
      int hidden_idx = i % length;
      
      #ifdef USING_DOUBLES
      DTYPE mean_i = 1.0 / (1.0 + exp(-(dot_product[i] + bias[hidden_idx])));
      #else
      DTYPE mean_i = 1.f / (1.f + exp(-(dot_product[i] + bias[hidden_idx])));
      #endif
      
      mean[i] = mean_i;
      
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_states[i]);//double mode
      #else
      DTYPE r = curand_uniform(&curand_states[i]);//float mode
      #endif
      
      sample[i] = r < mean_i;
    }
  }

  //Average time: 0.000766
  //Matrix based version of sample_h_given_v. Basically does
  //sample_h_given_v for all batch elements. 
  void RBM_dp_matrix::sample_h_given_v_matrix(DTYPE *dev_v0_sample, DTYPE *dev_mean,
					      DTYPE *dev_sample) {
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    gemm(dev_v0_sample,
#ifdef WEIGHT_MATRIX_PADDING
	 dev_W_cublas,
#else
	 dev_W,
#endif
	 dev_h_dot_product_batch,
	 batch_size, n_hidden, n_visible,
	 false, true, host_handle);
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    int num_blocks = ((n_hidden * batch_size - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    finish_sampling_kernel_matrix<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (dev_mean, dev_h_dot_product_batch, dev_sample, dev_hbias, n_hidden, dev_curand_states);
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
  }
  
  void RBM_dp_matrix::sample_v_given_h_matrix(DTYPE *h0_sample, DTYPE *v_mean,
					      DTYPE *v_sample, DTYPE *prev_v_sample) {
    // cudaDeviceSynchronize();//sync from sample_h_given_v_cublas
    // matrix_dot_vector(W, const_n_hidden, const_n_visible, h0_sample,
    // 		      v_dot_product, true, handle);
    gemm(h0_sample,
#ifdef WEIGHT_MATRIX_PADDING
	 dev_W_cublas,
#else
	 dev_W,
#endif
	 dev_v_dot_product_batch,
	 batch_size, n_visible, n_hidden,
	 false, false, host_handle);

    add_bias(v_mean, dev_v_dot_product_batch, dev_vbias, n_visible);
    save_sample_changes(v_mean, v_sample, prev_v_sample, dev_v_diffs, n_visible
			#ifndef BIT_CODING
			, vdirections
			#endif
			);
  }


#ifdef AGG_WARP_FILT

    //BEGIN: Code from url: "https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/"
  
  __device__
  inline int lane_id(void) {
    return threadIdx.x % WARP_SIZE;
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


  //length is the length of each dot product (probably either
  //n_visible or n_hidden), and padded_length is the length rounded up
  //to the nearest warp size.

  //At the end of this function we want the value at diffs[0] to
  //represent the number of diffs that follow in the array. So we can
  //just send the diffs ptr to atomicAggInc so that it modifies the
  //first element accordingly.  However, it's still really
  //complicated, since there are separate aggregations for each diffs
  //array in a batch happening simultaneously...  Warps are
  //indivisible, but we'd need them to start their relative count
  //within a subwarp. The easiest way to fix this would be to set the
  //block size to "length" with "batch_size" blocks. However, this
  //would not support any layer with more than 1024 elements since
  //that is the max threadperblock count in CUDA.  So to make it
  //general we round up length to the nearest divisible warp size when
  //calling this kernel, then perform some extra index shifting and
  //checking inside the kernel. As it turns out, the easy way to do
  //this is to allocate a 2D set of threads.
  //
  //WARNING: The diffs stored in the array will be the modded (non
  //batch) index, and this is for use later on in the dpmm kernel.
  __global__
  void save_sample_changes_kernel(DTYPE * mean, DTYPE * sample,
				  DTYPE * prev_sample,
				  int * diffs_batch, int length, 
				  curandState_t * curand_state_ptr
				  #ifndef BIT_CODING
				  , bool * directions_batch
				  #endif
				  ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int batch_i = blockDim.y * blockIdx.y + threadIdx.y;
    //Note: i no longer corresponds to the correct index for mean,
    //sample, and prev_sample due to the padding. The actual thread
    //indices will be calculated later in the kernel.
    if((i < length) && (batch_i < const_batch_size)) {
      //Only activate threads that are within a batch's length
      //Still need a shift for diffs array (cause it has a different
      //length)
      int shifted_i = (batch_i * length) + i;
      //Now we begin actual calculations
      DTYPE prev_sample_i = prev_sample[shifted_i];
      // DTYPE r = curand_uniform_DTYPE(&curand_state_ptr[shifted_i]);
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[shifted_i]);//double mode
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[shifted_i]);//float mode
      #endif

      DTYPE new_sample_i = r < mean[shifted_i] ? 1.f : 0.f;//Just to be sure it's 1 or 0
      sample[shifted_i] = new_sample_i;
      if(new_sample_i != prev_sample_i) {
	#ifdef BIT_CODING
	//Construct even number to indicate default dot product increasing
	int my_diff_value = 2 * i;
	if(prev_sample_i != 0.f) {
	  my_diff_value++;
	}
	#else
	int my_diff_value = i;
	#endif
	
	int diffs_shift = batch_i * (length+1);
	int * diffs = &diffs_batch[diffs_shift];
	int my_idx = atomicAggInc(diffs);
	//+1 to avoid first index (stores size)
	diffs[my_idx + 1] = my_diff_value;
	#ifndef BIT_CODING
	bool * directions = &directions_batch[batch_i * length];
	#ifdef USING_DOUBLES
	directions[my_idx] = (prev_sample_i == 0.0);
	#else
	directions[my_idx] = (prev_sample_i == 0.f);
	#endif
	#endif
      }
    }
  }
#else
  __global__
  void save_sample_changes_kernel(DTYPE * mean, DTYPE * sample,
				  DTYPE * prev_sample,
				  int * diffs_batch, int length, 
				  curandState_t * curand_state_ptr
#ifndef BIT_CODING
				  , bool * directions_batch
#endif
				  ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int batch_i = blockDim.y * blockIdx.y + threadIdx.y;
    //Note: i no longer corresponds to the correct index for mean,
    //sample, and prev_sample due to the padding. The actual thread
    //indices will be calculated later in the kernel.
    if((i < length) && (batch_i < const_batch_size)) {
      //Only activate threads that are within a batch's length
      //Still need a shift for diffs array (cause it has a different
      //length)
      int shifted_i = (batch_i * length) + i;
      //Now we begin actual calculations
      DTYPE prev_sample_i = prev_sample[shifted_i];
      // DTYPE r = curand_uniform_DTYPE(&curand_state_ptr[shifted_i]);
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[shifted_i]);//double mode
      DTYPE new_sample_i = r < mean[shifted_i] ? 1.0 : 0.0;//Just to be sure it's 1 or 0
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[shifted_i]);//float mode
      DTYPE new_sample_i = r < mean[shifted_i] ? 1.f : 0.f;//Just to be sure it's 1 or 0
      #endif
      
      sample[shifted_i] = new_sample_i;
      if(new_sample_i != prev_sample_i) {
	//Construct even number to indicate default dot product increasing
	#ifdef BIT_CODING
	int my_diff_value = 2 * i;//instead of shifted_i
	if(prev_sample_i != 0.0) {
	  //Then changing to negative, so dot product decreases, so
	  //change to odd number (this encodes which direction the
	  //dot product is going without extra memory usage)
	  my_diff_value++;
	}
	#else
	int my_diff_value = i;
	#endif
	int diffs_shift = batch_i * (length+1);
	int * diffs = &diffs_batch[diffs_shift];
	// int my_idx = atomicAggInc(diffs);
	int my_idx = atomicAdd(diffs, 1);
	//+1 to avoid first index (stores size)
	diffs[my_idx + 1] = my_diff_value;
	#ifndef BIT_CODING
	bool * directions = &directions_batch[batch_i * length];
	#ifdef USING_DOUBLES
	directions[my_idx] = (prev_sample_i == 0.0);
	#else
	directions[my_idx] = (prev_sample_i == 0.f);
	#endif
	#endif
      }
    }
  }  
#endif
    
  __global__
  void reset_diffs_counters(int * diffs, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < const_batch_size) {
      diffs[i * (length+1)] = 0;
    }
  }
  

  //Note: sample and prev_sample could be the same pointer.
  void RBM_dp_matrix::save_sample_changes(DTYPE * mean, DTYPE * sample,
					  DTYPE * prev_sample, int * diffs, int length
					  #ifndef BIT_CODING
					  , bool * directions
					  #endif
					  ) {
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    //Set initial diffs index counter to 0 for each batch
    reset_diffs_counters<<<((batch_size - 1) / 32) + 1, 32>>>(diffs, length);
    
    //Need to setup some special padding for this kernel
    // int padded_length = length + (length%WARP_SIZE);
    // int num_threads = batch_size * padded_length;
    // cout << "length: " << length << endl;
    // cout << "padded_length: " << padded_length << endl;
    // cout << "batch_size: " << batch_size << endl;
    // num_blocks = ((num_threads - 1) / SAVE_SAMPLE_CHANGES_KERNEL_BLOCK_SIZE) + 1;
    // cout << "num threads: " << num_threads << endl;
    // cout << "num_blocks: " << num_blocks << endl;
    dim3 num_blocks, num_threads;
    dims_to_num_threads_and_blocks(length, batch_size, num_blocks, num_threads);
    save_sample_changes_kernel<<<num_blocks, num_threads>>>
      (mean, sample, prev_sample, diffs, length, dev_curand_states
       #ifndef BIT_CODING
       ,directions
       #endif
       );
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    //Print the changes for testing...
    // int * host_diffs = new int[batch_size * (length+1)];
    // cudaMemcpy(host_diffs, diffs, batch_size * (length+1) * sizeof(int),
    // 	       cudaMemcpyDeviceToHost);
    // int indices[] = {0, 1};
    // for(int idx = 0; idx < 2; idx++) {
    //   int * my_diffs = &host_diffs[indices[idx] * (length+1)];
    //   cout << "idx: " << idx << endl;
    //   for(int i = 0; i < my_diffs[0]+3; i++) {
    // 	cout << my_diffs[i] << endl;
    //   }
    //   cout << endl;
    // }
    // delete[] host_diffs;
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
    if(i < length * const_batch_size) {
      int length_i = i % length;
#ifdef USING_DOUBLES
      mean[i] = 1.0 / (1.0 + exp(-(dot_product[i] + bias[length_i])));
#else
      mean[i] = 1.f / (1.f + exp(-(dot_product[i] + bias[length_i])));
#endif
    }
  }

  void RBM_dp_matrix::add_bias(DTYPE * mean, DTYPE * dot_product,
			       DTYPE * bias, int length) {
    int num_blocks = (((length*batch_size) - 1) / ADD_BIAS_KERNEL_BLOCK_SIZE) + 1;
    add_bias_kernel<<<num_blocks, ADD_BIAS_KERNEL_BLOCK_SIZE>>>
      (mean, dot_product, bias, length);
    // cudaDeviceSynchronize();
  }





  //len_diffs is the actual length of each array in diffs_batch
  __global__
  void dpmm_kernel(DTYPE * W, int m, int n, int * diffs_batch, int len_diffs,
		   DTYPE * dot_product_batch, int len_dot_product, bool transpose
		   #ifndef BIT_CODING
		   , bool * directions
		   #endif
		   ) {
    // extern __shared__ int diffs_batch_sh[];
    // __shared__ int result_inc[1024];//TODO
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int batch_i = blockDim.y * blockIdx.y + threadIdx.y;
    //TODO: Implement a universal diffs transfer to more efficient
    //memory so that accessing diffs takes less time...
    if((i < len_dot_product) && (batch_i < const_batch_size)) {
      //Setup shifted pointers
      DTYPE * dot_product = &dot_product_batch[batch_i * len_dot_product];
      DTYPE result_i_inc = 0.0f;
      int num_diffs = diffs_batch[batch_i * len_diffs];

      // if(num_diffs > len_dot_product) printf("PROBLEM\n");
      // if(i < num_diffs+1) {//WARNING: Need to fix potential missing threads
      // 	diffs_batch_sh[i] = diffs_batch[(batch_i * len_diffs) + i];
      // }
      int my_base = (batch_i * len_diffs);
      int my_dir_base = (batch_i * (len_diffs-1)) - 1;
      for(int diffs_idx = 1; diffs_idx < num_diffs+1; diffs_idx++) {
	int curr_diff = diffs_batch[my_base + diffs_idx];
	// int curr_diff = diffs_batch_sh[diffs_idx];
	#ifdef BIT_CODING
	int diff_value = curr_diff >> 1;
	bool diff_direction = (curr_diff & 1) == 0;
        #else
	int diff_value = curr_diff;
	bool diff_direction = directions[my_dir_base + diffs_idx];
	#endif 
	// int curr_diff = diffs_sh[diffs_idx-1];
	int idx;
	//
	//ASSERT: m = n_hidden, n = n_visible
#ifdef MULTI_WEIGHT_MATRIX
	
#ifdef WEIGHT_MATRIX_PADDING
	if(transpose) //Then W2 is being used
	  idx = diff_value * (const_pitch2/sizeof(DTYPE)) + i;
	else
	  idx = diff_value * (const_pitch/sizeof(DTYPE)) + i;
#else
	if(transpose)
	  idx = diff_value * m + i;
	else
	  idx = diff_value * n + i;
#endif
	
#else

#ifdef WEIGHT_MATRIX_PADDING
	if(transpose)
	  idx = i * (const_pitch/sizeof(DTYPE)) + diff_value;
	else
	  idx = diff_value * (const_pitch/sizeof(DTYPE)) + i;
#else
	if(transpose)
	  idx = i * n + diff_value;
	else
	  idx = diff_value * n + i;
#endif
	
#endif
	
	// result_i_inc += W[idx] * (curr_diff % 2 == 0 ? 1.0 : -1.0);
	// if(diff_direction)
	//   result_i_inc += W[idx];
	// else
	//   result_i_inc -= W[idx];
	
	result_i_inc += W[idx] * (diff_direction ? 1 : -1);
	// result_inc[i] += W[idx] * (diff_direction ? 1 : -1);
      }
      dot_product[i] += result_i_inc;
    }
  }


#ifndef USE_DPMM
  //num blocks in y dimension, must be >=3 for float to limit shared memory, >=6 for doubles (probably)
  #define DPVM_Y_NUM_BLOCKS 3
  
  //This determines the number of blocks in the z dimension.
  //Increasing yields more parallelism, but less shared memory reuse. 
  #define NUM_BATCH_BLOCKS 16
  
  
  __global__
  void dpvm_kernel(DTYPE * W, int m, int n, int * diffs_batch, int len_diffs,
		   DTYPE * dot_product_batch, int len_dot_product, bool transpose
		   #ifndef BIT_CODING
		   , bool * directions_batch
		   #endif
		   ) {
    __shared__ float block_results[WARP_SIZE];
    __shared__ DTYPE W_sh[WARP_SIZE * (784 / DPVM_Y_NUM_BLOCKS)];//diffs index the W matrix's rows (len_diffs-1), TODO: upgrade to dynamic to fix
    int W_col_idx = blockDim.x * blockIdx.x + threadIdx.x;//Used for weight matrix accessing
    //Transfer all W data to shared memory (can this be upgraded to conditional transfer if used?).
    int num_W_rows_per_block = (len_diffs - 1) / DPVM_Y_NUM_BLOCKS;
    for(int i_iter = 0; i_iter < num_W_rows_per_block; i_iter++) {
      int i = blockIdx.y * num_W_rows_per_block + i_iter;//i is the row
      if((i < len_diffs - 1) && (W_col_idx < len_dot_product)) {
	W_sh[i_iter * WARP_SIZE + threadIdx.x] = W[i * (transpose ? m:n) + W_col_idx];
      }
    }
    // __syncthreads();

    // int y = threadIdx.y;
    // int block_results_idx = threadIdx.y * blockDim.x + threadIdx.x;//Used later as well
    int block_results_idx = threadIdx.x;//Used later as well
    
    //setup ptrs for specific batch and iterate
    int num_batch_els = const_batch_size / NUM_BATCH_BLOCKS;
    //an iter (batch_iter) is the iteration index we're on, the index (batch_i) is the batch element! Important distinction here
    for(int batch_iter = 0; batch_iter < num_batch_els; batch_iter++) {
      int batch_i = blockIdx.z * num_batch_els + batch_iter;
      
      int * diffs = &diffs_batch[len_diffs * batch_i];
      DTYPE * dot_product = &dot_product_batch[len_dot_product * batch_i];
      bool * directions = &directions_batch[(len_diffs-1) * batch_i];
      
      //Reset the results shared mem
      block_results[block_results_idx] = 0.f;
      
      //TODO: Implement a universal diffs transfer to more efficient
      //memory so that accessing diffs takes less time...
      if(W_col_idx < len_dot_product) {//TODO: consider moving up
	//Setup shifted pointers
	int num_diffs = diffs[0];
	// int num_diffs_iters = ((num_diffs-1) / DPVM_Y_BLOCK_SIZE) + 1 + 1;
	//+1 for normal blocking calculation, +1 for diffs array shift
	for(int diffs_idx = 1; diffs_idx < num_diffs+1; diffs_idx++) {
	  // int diffs_idx = diffs_base_idx + threadIdx.y;
	  // if(diffs_idx - 1 < num_diffs) {
	  int curr_diff = diffs[diffs_idx];
#ifdef BIT_CODING
	  int diff_value = curr_diff >> 1;
	  bool diff_direction = (curr_diff & 1) == 0;
#else
	  int diff_value = curr_diff;
	  bool diff_direction = directions[diffs_idx-1];
#endif
	  //Lots of unnecessary checks, but whatevs
	  if((blockIdx.y * num_W_rows_per_block <= diff_value) && (diff_value < (blockIdx.y+1) * num_W_rows_per_block)) {
	    // int curr_diff = diffs_sh[diffs_idx-1];
	    int idx;
	    //
	    //ASSERT: m = n_hidden, n = n_visible
#ifdef MULTI_WEIGHT_MATRIX
	
#ifdef WEIGHT_MATRIX_PADDING
	    if(transpose) //Then W2 is being used
	      idx = diff_value * (const_pitch2/sizeof(DTYPE)) + i;
	    else
	      idx = diff_value * (const_pitch/sizeof(DTYPE)) + i;
#else
	    // if(transpose)
	    //   idx = diff_value * m + i;
	    // else
	    // idx = diff_value * n + i;
	    // if(transpose)
	    idx = (diff_value - (blockIdx.y * num_W_rows_per_block)) * WARP_SIZE + threadIdx.x;
	    // else
	    //   idx = diff_value * WARP_SIZE
#endif
	
#else

#ifdef WEIGHT_MATRIX_PADDING
	    if(transpose)
	      idx = i * (const_pitch/sizeof(DTYPE)) + diff_value;
	    else
	      idx = diff_value * (const_pitch/sizeof(DTYPE)) + i;
#else
	    if(transpose)
	      idx = i * n + diff_value;
	    else
	      idx = diff_value * n + i;
#endif
	
#endif
	
	    // result_i_inc += W[idx] * (diff_direction ? 1 : -1);
	    // block_results[block_results_idx] += W[idx] * (diff_direction ? 1 : -1);
	    block_results[block_results_idx] += W_sh[idx] * (diff_direction ? 1 : -1);
	    // }
	  }
	}
	atomicAdd(&dot_product[W_col_idx], block_results[block_results_idx]);
	// __syncthreads();
	// if(threadIdx.y == 0) {
	  /*
	    for(int k = 0; k < DPVM_Y_BLOCK_SIZE; k++) {
	    dot_product[i] += block_results[k * blockDim.x + threadIdx.x];
	    }//incrementing in shared memory isn't faster, I guess it's not a bottleneck.
	  */
	  /*
	  for(int k = 1; k < DPVM_Y_BLOCK_SIZE; k++) {
	    block_results[threadIdx.x] += block_results[k * blockDim.x + threadIdx.x];
	  }
	  dot_product[W_col_idx] += block_results[threadIdx.x];
	  */
	  
	// }
	// __syncthreads();//blocks could race ahead and reset block_results before being read
      }
      
      /*
      //Parallel reduce to first row of shared memory for the block
      int num_reducing = DPVM_Y_BLOCK_SIZE / 2;
      int access_shift = 1;
      while(num_reducing != 0) {
	if(threadIdx.y % (DPVM_Y_BLOCK_SIZE / num_reducing) == 0)
	  block_results[block_results_idx] += block_results[block_results_idx + (access_shift * blockDim.x)];
	num_reducing /= 2;
	access_shift *= 2;
	__syncthreads();
      }
      //Send to global memory
      if(threadIdx.y == 0)
	dot_product[i] += block_results[threadIdx.x];
      */
    }
  }
#endif
  
  void RBM_dp_matrix::dpmm(int m, int n, int * diffs,
			   DTYPE * dot_product, bool transpose
			   #ifndef BIT_CODING
			   , bool * directions
			   #endif
			   ) {
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    //Figure out the correct length of the dot_product based on transpose:
    int len_dot_product = transpose ? m : n;
    int len_diffs       = transpose ? n+1 : m+1;//diffs arrays are longer

    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    #ifdef MULTI_WEIGHT_MATRIX
    DTYPE * corrected_W = transpose ? dev_W2 : dev_W;
    #else
    DTYPE * corrected_W = dev_W;
    #endif

#ifdef USE_DPMM
    dim3 num_blocks(getNumBlocks(len_dot_product, WARP_SIZE),
		    getNumBlocks(batch_size, DPMM_Y_BLOCK_SIZE));
    dim3 num_threads(WARP_SIZE, DPMM_Y_BLOCK_SIZE);
    
    dpmm_kernel<<<num_blocks, num_threads>>> (corrected_W, m, n, diffs, len_diffs,
					      dot_product, len_dot_product, transpose
					      #ifndef BIT_CODING
					      , directions
					      #endif
					      );
#else
    // int num_blocks = getNumBlocks(len_dot_product, WARP_SIZE);
    // int num_threads = WARP_SIZE;
    dim3 num_blocks(getNumBlocks(len_dot_product, WARP_SIZE), DPVM_Y_NUM_BLOCKS, NUM_BATCH_BLOCKS);
    dim3 num_threads(WARP_SIZE, 1, 1);//only 1 z thd, since it will iterate
    dpvm_kernel<<<num_blocks, num_threads>>> (corrected_W, m, n, diffs, len_diffs,
					      dot_product, len_dot_product, transpose
#ifndef BIT_CODING
					      , directions
#endif
					      );
    /*
    dim3 num_blocks(getNumBlocks(len_dot_product, WARP_SIZE), 1);
    dim3 num_threads(WARP_SIZE, DPVM_Y_BLOCK_SIZE);
    for(int i = 0; i < batch_size; i++) {
      dpvm_kernel<<<num_blocks, num_threads>>> (corrected_W, m, n, diffs + (len_diffs * i), len_diffs,
						dot_product + (len_dot_product * i),
						len_dot_product, transpose
#ifndef BIT_CODING
						, directions + (i * (len_diffs - 1))
#endif
						);
    }
    */
#endif
    // cudaDeviceSynchronize();
    // double t2 = get_wall_time();
    // cout << "dpmm microseconds: " << (t2-t1) * 1000000 << endl;

    //doubles
    // dpmm: 0.000123024
    // dpmm: 0.000359058
    //floats
    // dpmm: 0.00114083 <<< 10X slower?
    // dpmm: 0.000381947

    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
  }
  
  //PRE: prev_sample should be the last sample array that was sent
  //     into a sample_h_given_v* call.
  //TODO: This one is taking a long time (around 0.9-1.4 ms)
  //      s_v_g_h_delta only takes 0.38-0.39ms
  //      Room for improvement!
  void RBM_dp_matrix::sample_h_given_v_delta(DTYPE * v0_sample, DTYPE * h_mean,
					     DTYPE * h_sample, DTYPE * prev_h_sample) {
    // cerr << "sample_h_given_v_delta\n";
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    //This code performs a column analysis and prints if there's a column with no diffs. 
    int * v_diffs = new int[(n_visible+1) * batch_size];
    cudaMemcpy(v_diffs, dev_v_diffs, (n_visible+1) * batch_size * sizeof(int),
	       cudaMemcpyDeviceToHost);
    bool * v_checked = new bool[n_visible];
    for(int i = 0; i < n_visible; i++) {
      v_checked[i] = true;
    }
    for(int row = 0; row < batch_size; row++) {
      int num_diffs = v_diffs[row * (n_visible+1)];
      // cout << "num_diffs = " << num_diffs << endl;
      for(int i = 1; i < num_diffs+1; i++) {
	int col = v_diffs[row * (n_visible+1) + i];
	v_checked[col] = false;
      }
    }

    int num_cols_empty = 0;
    for(int i = 0; i < n_visible; i++) {
      if(v_checked[i]) {
	num_cols_empty++;
	// cout << i << " is EMPTY" << endl;
      }
    }
    // cout << "percent cols empty = " << (num_cols_empty / ((float) n_visible)) << endl;
    delete[] v_diffs;
    delete[] v_checked;
    
    // int num_diffs_first_v;
    // cudaMemcpy(&num_diffs_first_v, dev_v_diffs, sizeof(int), cudaMemcpyDeviceToHost);
    // cout << "v percent diff: " << (num_diffs_first_v/((float)n_visible)) << endl;
    // cout << ": " << (num_diffs_first_v/((float)n_visible)) << endl;
    

    dpmm(n_hidden, n_visible, dev_v_diffs, dev_h_dot_product_batch, true
	 #ifndef BIT_CODING
	 , vdirections
	 #endif
	 );
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    add_bias(h_mean, dev_h_dot_product_batch, dev_hbias, n_hidden);

    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    save_sample_changes(h_mean, h_sample, prev_h_sample, dev_h_diffs, n_hidden
#ifndef BIT_CODING
			, hdirections
#endif
			);
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
  }


  //PRE: prev_sample should be the last sample array that was sent
  //     into a sample_v_given_h* call.
  //     All paramters should be device allocated.
  void RBM_dp_matrix::sample_v_given_h_delta(DTYPE * h0_sample, DTYPE * v_mean,
					     DTYPE * v_sample, DTYPE * prev_v_sample) {
    // cerr << "sample_v_given_h_delta\n";
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    //TODO: Fix MULTI_WEIGHT_MATRIX
    // cerr << "n_hidden: " << n_hidden << endl;
    // cerr << "n_visible: " << n_visible << endl;
    // int num_diffs_first_h;
    // cudaMemcpy(&num_diffs_first_h, dev_h_diffs, sizeof(int), cudaMemcpyDeviceToHost);
    // cout << "h percent diff: " << (num_diffs_first_h/((float)n_hidden)) << endl;

    
    dpmm(n_hidden, n_visible, dev_h_diffs, dev_v_dot_product_batch, false
	 #ifndef BIT_CODING
	 , hdirections
	 #endif
	 );
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    add_bias(v_mean, dev_v_dot_product_batch, dev_vbias, n_visible);
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    save_sample_changes(v_mean, v_sample, prev_v_sample, dev_v_diffs, n_visible
#ifndef BIT_CODING
			, vdirections
#endif
			);
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
  }


// #ifdef MULTI_WEIGHT_MATRIX
  // __global__
  // void write_matrix_transpose(DTYPE * W, DTYPE * W2) {
  //   int x = blockIdx.x * blockDim.x + threadIdx.x;
  //   int y = blockIdx.y * blockDim.y + threadIdx.y;
  //   __shared__ DTYPE my_tile[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE];
  //   if((x < const_n_visible) && (y < const_n_hidden)) {
  //     my_tile[threadIdx.y][threadIdx.x] = W[y * const_n_visible + x];
  //   }
  //   __syncthreads();
  //   if((y < const_n_visible) && (x < const_n_hidden)) {
      
  //   }
  // }

  // __global__ void write_matrix_transpose(float *W, const float *W2)
  // {
  //   __shared__ float tile[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE+1];

  //   int x = blockIdx.x * MAX_THREAD_SQUARE_EDGE + threadIdx.x;
  //   int y = blockIdx.y * MAX_THREAD_SQUARE_EDGE + threadIdx.y;
  //   int width = gridDim.x * MAX_THREAD_SQUARE_EDGE;

  //   for (int j = 0; j < MAX_THREAD_SQUARE_EDGE; j += BLOCK_ROWS)
  //     tile[(threadIdx.y+j)*MAX_THREAD_SQUARE_EDGE + threadIdx.x] = W[(y+j)*width + x];

  //   __syncthreads();

  //   for (int j = 0; j < MAX_THREAD_SQUARE_EDGE; j += BLOCK_ROWS)
  //     W2[(y+j)*width + x] = tile[(threadIdx.y+j)*MAX_THREAD_SQUARE_EDGE + threadIdx.x];
  // }
  
// #endif
  
  // #define TILE_DIM 32
  //PRE: src has const_n_hidden rows and const_n_visible columns.
  //     dest is allocated with const_n_hidden * const_n_visible elements
  //POST: Writes the tranpose of src to dest.
  // __global__
  // void write_matrix_transpose(DTYPE * src, DTYPE * dest) {
  //   __shared__ float tile[MAX_THREAD_SQUARE_EDGE][MAX_THREAD_SQUARE_EDGE];

  //   int x = blockIdx.x * blockDim.x + threadIdx.x;
  //   int y = blockIdx.y * blockDim.y + threadIdx.y;
  //   if((x < const_n_visible) && (y < const_n_hidden)) {
  //     tile[threadIdx.y][threadIdx.x] = src[y*const_n_visible + x];
  //   }
  //   __syncthreads();
  //   x = blockIdx.y * blockDim.y + threadIdx.x;  // transpose block offset
  //   y = blockIdx.x * blockDim.x + threadIdx.y;
  //   if((x < const_n_hidden) && (y < const_n_visible)) {
  //     dest[y*const_n_hidden + x] = tile[threadIdx.x][threadIdx.y];
  //   }
  // }

  void RBM_dp_matrix::gibbs_hvh_delta(DTYPE *h0_sample, DTYPE *nv_means, DTYPE *nv_samples,
				      DTYPE *nh_means, DTYPE *nh_samples) {
    sample_v_given_h_delta(h0_sample , nv_means, nv_samples, nv_samples);
    sample_h_given_v_delta(nv_samples, nh_means, nh_samples, nh_samples);
  }

  DTYPE abs(DTYPE d) {
    if(d < 0) return -d;
    return d;
  }

  void compare_W_W2(DTYPE * dev_W, DTYPE * dev_W2, int n_visible, int n_hidden) {
    DTYPE * W = new DTYPE[n_visible * n_hidden];
    DTYPE * W2 = new DTYPE[n_visible * n_hidden];
    cudaMemcpy(W, dev_W, sizeof(DTYPE) * n_visible * n_hidden, cudaMemcpyDeviceToHost);
    cudaMemcpy(W2, dev_W2, sizeof(DTYPE) * n_visible * n_hidden, cudaMemcpyDeviceToHost);
    DTYPE total_diff = 0.0;
    for(int row = 0; row < n_hidden; row++) {
      for(int col = 0; col < n_visible; col++) {
  	total_diff += abs(W[row * n_visible + col] - W2[col*n_hidden + row]);
      }
    }
    cout << "W diff: " << total_diff << endl;
    delete[] W;
    delete[] W2;
  }
  
  void RBM_dp_matrix::contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc,
					     DTYPE * dev_data) {
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    //TODO: Upgrade to use the multi-weight matrix functionality
    // reset_d_arrays();
    
    DTYPE * dev_input = &dev_data[data_num_cols * (curr_i*batch_size)];
    sample_h_given_v_matrix(dev_input          , dev_ph_mean_batch , dev_ph_sample_batch);
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    sample_v_given_h_matrix(dev_ph_sample_batch, dev_nv_means_batch, dev_nv_samples_batch,
			    dev_input);//include dev_input as the
				       //previous sample
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    sample_h_given_v_delta(dev_nv_samples_batch, dev_nh_means_batch, dev_nh_samples_batch,
    			   dev_ph_sample_batch);
    for(int step = 1; step < k; step++) {
      // sample_v_given_h_delta(dev_nh_samples_batch, dev_nv_means_batch, dev_nv_samples_batch,
      // 			     dev_nv_samples_batch);
      // cerr << "sample_h_given_v_delta\n";
      // sample_h_given_v_delta(dev_nv_samples_batch, dev_nh_means_batch, dev_nh_samples_batch,
      // 			     dev_nh_samples_batch);
      // cerr << "gibbs_hvh_delta\n";
      gibbs_hvh_delta(dev_nh_samples_batch, dev_nv_means_batch, dev_nv_samples_batch,
      		      dev_nh_means_batch, dev_nh_samples_batch);
    }
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    int most_nodes = max(n_hidden, n_visible);
    
    dim3 num_blocks, num_threads;
    dims_to_num_threads_and_blocks(n_visible, n_hidden, num_blocks, num_threads);
    
    // #ifdef SIMULTANEOUS_EXECUTION
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    // GET_TIME(t1);
    //MULTI_WEIGHT_MATRIX only adds 0.00035 seconds extra time onto
    //   what was originally 0.00195 seconds. (GTX 980 test)
    write_weights <<< num_blocks, num_threads>>>
      (dev_data, dev_W,
#ifdef MULTI_WEIGHT_MATRIX
       dev_W2,
#endif
       lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);
    // #ifdef MULTI_WEIGHT_MATRIX
    // write_transpose<<< num_blocks, num_threads>>> (dev_W, dev_W2);
    // #endif
#ifdef MULTI_WEIGHT_MATRIX
    #ifdef EFFICIENT_TRANSPOSE
    dim3 dimGrid((n_visible-1)/MAX_THREAD_SQUARE_EDGE+1, (n_hidden-1)/MAX_THREAD_SQUARE_EDGE+1);
    dim3 dimBlock(MAX_THREAD_SQUARE_EDGE, BLOCK_ROWS);
    
    //Need to update W2 efficiently
    // dims_to_num_threads_and_blocks(most_nodes, most_nodes, num_blocks, num_threads);
    #ifdef WEIGHT_MATRIX_PADDING
    write_matrix_transpose_pitch<<<dimGrid, dimBlock>>>(dev_W, dev_W2);
    #else
    write_matrix_transpose<<<dimGrid, dimBlock>>>(dev_W, dev_W2);
    #endif
    #endif
#endif
    
    #ifdef WEIGHT_MATRIX_PADDING
    //Cant go device to device, so have to go hacky and copy it out
    //to main memory then back
    CUDA_CHECK(cudaMemcpy2D(WArray, n_visible * sizeof(DTYPE), dev_W, pitch,
			    n_visible * sizeof(DTYPE), n_hidden,
			    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dev_W_cublas, WArray, n_visible * n_hidden * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy2D(dev_W_cublas, n_visible * sizeof(DTYPE), dev_W, pitch,
    // 			    n_visible * sizeof(DTYPE), n_hidden * sizeof(DTYPE),
    // 			    cudaMemcpyDeviceToDevice));
    #endif
    
    // compare_W_W2(dev_W, dev_W2, n_visible, n_hidden);
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    // GET_TIME(t2);
    // cerr << "weight matrix update time: " << get_duration(t1, t2) << endl;

    int num_bias_blocks = 1 + ((most_nodes-1) / WRITE_BIAS_KERNEL_BLOCK_SIZE);
    int num_bias_threads = WRITE_BIAS_KERNEL_BLOCK_SIZE;
    write_bias_results_to_memory <<< num_bias_blocks, num_bias_threads>>>
      (dev_data, lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);


    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    // GET_TIME(k2_t2);
    // cerr << "k2 time: " << get_duration(k2_t1, k2_t2) << endl;
  }

  
  void RBM_dp_matrix::reset_d_arrays() {
    //Since dW_pitch is the width of the dev_dW array rows, we
    //multiply by the number of rows (n_hidden) to get the number of
    //bytes to reset:
    // CUDA_CHECK(cudaMemset(dev_dhbias, 0, n_hidden  * sizeof(DTYPE)));
    // CUDA_CHECK(cudaMemset(dev_dvbias, 0, n_visible * sizeof(DTYPE)));
    
    // CUDA_CHECK(cudaMemset(dev_ph_mean_batch   , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nv_means_batch  , 0, sizeof(DTYPE) * n_visible * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nh_means_batch  , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    // CUDA_CHECK(cudaMemset(dev_ph_sample_batch , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nv_samples_batch, 0, sizeof(DTYPE) * n_visible * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nh_samples_batch, 0, sizeof(DTYPE) * n_hidden  * batch_size));
  }


  void RBM_dp_matrix::allocate_special_memory() {
    // data = new DTYPE[data_num_rows * data_num_cols];
    // for(int i = 0; i < data_num_rows * data_num_cols; i++) {
    //   data[i] = (DTYPE) int_data[i];
    // }
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc((void**)&dev_ph_sample_batch , sizeof(DTYPE) * n_hidden  * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nv_samples_batch, sizeof(DTYPE) * n_visible * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nh_samples_batch, sizeof(DTYPE) * n_hidden  * batch_size));

    #ifdef WEIGHT_MATRIX_PADDING
    CUDA_CHECK(cudaMallocPitch((void**)&dev_W, &pitch, n_visible * sizeof(DTYPE), n_hidden));
    //Copy pitch to const memory
    cudaMemcpyToSymbol(const_pitch, &pitch , sizeof(size_t));
    matrixToArray(W, WArray , n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy2D(dev_W, pitch, WArray, n_visible * sizeof(DTYPE),
			    n_visible * sizeof(DTYPE), n_hidden,
			    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&dev_W_cublas, n_hidden * n_visible * sizeof(DTYPE)));
    // CUDA_CHECK(cudaMemcpy2D(dev_W_cublas, n_visible * sizeof(DTYPE), dev_W, pitch,
    // 			    n_visible * sizeof(DTYPE), n_hidden * sizeof(DTYPE),
    // 			    cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dev_W_cublas, WArray, n_hidden * n_visible * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    
    #else
    CUDA_CHECK(cudaMalloc((void**)&dev_W , n_hidden * n_visible * sizeof(DTYPE)));
    matrixToArray     (W, WArray , n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(dev_W, WArray, n_hidden * n_visible * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    #endif

    //Allocate transpose(s)
    
    #ifdef MULTI_WEIGHT_MATRIX
    #ifdef WEIGHT_MATRIX_PADDING
    CUDA_CHECK(cudaMallocPitch((void**)&dev_W2, &pitch2, n_hidden * sizeof(DTYPE), n_visible));
    cudaMemcpyToSymbol(const_pitch2, &pitch2 , sizeof(size_t));
    matrixToArrayTrans(W, WArray2, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy2D(dev_W2, pitch2, WArray2, n_hidden * sizeof(DTYPE),
			    n_hidden * sizeof(DTYPE), n_visible,
			    cudaMemcpyHostToDevice));
    #else
    CUDA_CHECK(cudaMalloc((void**)&dev_W2, n_hidden * n_visible * sizeof(DTYPE)));
    matrixToArrayTrans(W, WArray2, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(dev_W2, WArray2, n_hidden * n_visible * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    #endif
    #endif
    
    // CUDA_CHECK(cudaMalloc((void**)&dev_data,
    // 			  data_num_rows * data_num_cols * sizeof(DTYPE)));
    // CUDA_CHECK(cudaMemcpy(dev_data, data, data_num_rows * data_num_cols * sizeof(DTYPE),
    // 			  cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&dev_h_dot_product_batch,
			  sizeof(DTYPE) * n_hidden * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_v_dot_product_batch,
			  sizeof(DTYPE) * n_visible  * batch_size));

    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

  }

  void RBM_dp_matrix::copy_matrices_to_host() {
#ifdef WEIGHT_MATRIX_PADDING
    CUDA_CHECK(cudaMemcpy2D(WArray, n_visible * sizeof(DTYPE), dev_W, pitch, 
			    n_visible * sizeof(DTYPE), n_hidden,
			    cudaMemcpyDeviceToHost));
#else
    CUDA_CHECK(cudaMemcpy(WArray, dev_W, n_hidden * n_visible * sizeof(DTYPE), 
    			  cudaMemcpyDeviceToHost));
#endif
    arrayToMatrix(WArray, W, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(vbias, dev_vbias, n_visible * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbias, dev_hbias, n_hidden  * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));

    
#ifdef DEBUG
    DTYPE * nh_samples = new DTYPE[n_hidden * batch_size];
    DTYPE * nv_samples = new DTYPE[n_visible * batch_size];
    DTYPE * nh_means = new DTYPE[n_hidden * batch_size];
    DTYPE * nv_means = new DTYPE[n_visible * batch_size];
    DTYPE * ph_means = new DTYPE[n_hidden * batch_size];
    DTYPE * ph_samples = new DTYPE[n_visible * batch_size];
    cudaMemcpy(nh_samples, dev_nh_samples_batch, n_hidden * batch_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(nv_samples, dev_nv_samples_batch, n_visible * batch_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(nh_means, dev_nh_means_batch, n_hidden * batch_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(nv_means, dev_nv_means_batch, n_visible * batch_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(ph_samples, dev_ph_sample_batch, n_hidden * batch_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(ph_means  , dev_ph_mean_batch, n_hidden * batch_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    
    int count = 5;

    cout << "nh_samples:\n";
    for(int i = 0; i < count; i++) {
      cout << nh_samples[i] << endl;
    }
    cout << "...\n";

    cout << "nv_samples:\n";
    for(int i = 0; i < count; i++) {
      cout << nv_samples[i] << endl;
    }
    cout << "...\n";

    cout << "nh_means:\n";
    for(int i = 0; i < count; i++) {
      cout << nh_means[i] << endl;
    }
    cout << "...\n";
    
    cout << "nv_means:\n";
    for(int i = 0; i < count; i++) {
      cout << nv_means[i] << endl;
    }
    cout << "...\n";
    
    cout << "ph_samples:\n";
    for(int i = 0; i < count; i++) {
      cout << ph_samples[i] << endl;
    }
    cout << "...\n";

    cout << "ph_means:\n";
    for(int i = 0; i < count; i++) {
      cout << ph_means[i] << endl;
    }
    cout << "...\n";

    delete[] nh_samples;
    delete[] nv_samples;
    delete[] nh_means;
    delete[] nv_means;
    delete[] ph_means;
    delete[] ph_samples;
#endif
  }

  RBM_dp_matrix::~RBM_dp_matrix() {
    cublasDestroy(host_handle);

#ifdef SAVE_WEIGHTS
    saveWeightMatrix();
#endif
    cudaFree(dev_h_diffs);
    cudaFree(dev_v_diffs);
    cudaFree(dev_h_dot_product_batch);
    cudaFree(dev_v_dot_product_batch);
    // destroy_cublas_handle<<<1,1>>>(dev_handle);
    // cudaFree(dev_handle);
    
    #ifdef MULTI_WEIGHT_MATRIX
    delete[] WArray2;
    cudaFree(dev_W2);
    #endif

    #ifdef WEIGHT_MATRIX_PADDING
    cudaFree(dev_W_cublas);
    #endif
    
    #ifndef BIT_CODING
    cudaFree(hdirections);
    cudaFree(vdirections);
    #endif
  }
}
