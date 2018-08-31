#include <iostream>
#include "../include/rbm_dynamic_cublas.h"
#include "../include/rbm_baseline.h"
#include "../include/utils.h"
#include "cublas_v2.h"
#include "../include/constants.h"

using namespace std;

#define SAMPLING_KERNEL_BLOCK_SIZE 32

namespace dynamic_cublas {

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
    		n, m,
    		&alpha,
    		matrix, n,
    		vector, 1,
    		&beta,
    		result, 1);
    #else
    cublasSgemv(*handle, transpose ? CUBLAS_OP_N : CUBLAS_OP_T,
    		n, m,
    		&alpha,
    		matrix, n,
    		vector, 1,
    		&beta,
    		result, 1);
    #endif
  }
  
  //This global function is designed to do 3 things:
  //1: Basic vector addition to add bias to mean.
  //2: Set mean[i] = sigmoid(mean[i])
  //3: Run random sampling and store in sample.
  //PRE: len(mean) == len(sample) == len(bias) == length
  __global__ void finish_sampling_kernel(DTYPE * mean, DTYPE * sample, DTYPE * bias,
					 int length, curandState_t * curand_state_ptr) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length) {
      DTYPE mean_i = 1.0 / (1.0 + exp(-(mean[i] + bias[i])));
      mean[i] = mean_i;
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[i]);
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[i]);
      #endif
      sample[i] = r < mean_i;
    }
  }
  
  __device__ void sample_h_given_v(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample,
				   DTYPE * W, DTYPE * hbias, 
				   curandState_t * curand_state_ptr, cublasHandle_t * handle) {
    //Goal is to compute (W . v0_sample) + hbias and store in mean,
    //then do the random number thing and store in sample
    matrix_dot_vector(W, const_n_hidden, const_n_visible, v0_sample, mean, false, handle);
    
    
    int num_blocks = ((const_n_hidden - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    finish_sampling_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (mean, sample, hbias, const_n_hidden, curand_state_ptr);
    cudaDeviceSynchronize();
  }

  //1: perform operation Transpose(W) . h0_sample
  //2: Run finish_sampling_kernel to do the rest of the operations. 
  __device__ void sample_v_given_h(DTYPE *h0_sample, DTYPE *mean, DTYPE *sample,
				   DTYPE * W, DTYPE * vbias, 
				   curandState_t * curand_state_ptr, cublasHandle_t * handle) {
    matrix_dot_vector(W, const_n_hidden, const_n_visible, h0_sample,
		      mean, true, handle);
    // cudaDeviceSynchronize();
    // if(blockIdx.x * blockDim.x + threadIdx.x == 0) {
    //   printf("first 10 mean\n");
    //   for(int i = 0; i < 10; i++) {
    // 	printf("%f\n", mean[i]);
    //   }
    // }

    int num_blocks = ((const_n_visible - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    finish_sampling_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (mean, sample, vbias, const_n_visible, curand_state_ptr);
  }
  
  __device__ void gibbs_hvh(DTYPE *h0_sample, DTYPE *nv_means,
			    DTYPE *nv_samples, DTYPE *nh_means,
			    DTYPE *nh_samples, DTYPE * W,
 			    DTYPE * hbias, DTYPE * vbias, 
			    curandState_t * curand_state_ptr, cublasHandle_t * handle) {
    sample_v_given_h(h0_sample, nv_means, nv_samples, W, vbias,
		     curand_state_ptr, handle);
    sample_h_given_v(nv_samples, nh_means, nh_samples, W, hbias,
		     curand_state_ptr, handle);
  }

  __global__ void cd_gpu(DTYPE * data, int curr_i, int data_num_cols, DTYPE * W,
			 DTYPE * hbias, DTYPE * vbias, curandState_t * curand_states,
			 DTYPE * ph_mean_batch, DTYPE * nv_means_batch,
			 DTYPE * nh_means_batch, DTYPE * ph_sample_batch,
			 DTYPE * nv_samples_batch, DTYPE * nh_samples_batch,
			 int curand_batch_width, cublasHandle_t * handle) {
    int batch_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(batch_i < const_batch_size) {
      DTYPE * ph_mean  = &ph_mean_batch[batch_i * const_n_hidden];
      DTYPE * nv_means = &nv_means_batch[batch_i * const_n_visible];
      DTYPE * nh_means = &nh_means_batch[batch_i * const_n_hidden];
      DTYPE * ph_sample   = &ph_sample_batch[batch_i * const_n_hidden];
      DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
      DTYPE * nh_samples  = &nh_samples_batch[batch_i * const_n_hidden];
      // if(batch_i == 0)
      // 	printf("hmean[0] = %f\n", nh_means[0]);
      
      curandState_t * curand_state_ptr = &curand_states[batch_i * curand_batch_width];
      DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
      
      sample_h_given_v(input, ph_mean, ph_sample, W, hbias, curand_state_ptr, handle);
      // printf("ph_mean[0], ph_mean[1] = %f, %f\n", ph_mean[0], ph_mean[1]);
      
      //Gibbs hvh but with different parameters
      sample_v_given_h(ph_sample, nv_means, nv_samples, W, vbias,
		       curand_state_ptr, handle);
      
      sample_h_given_v(nv_samples, nh_means, nh_samples, W, hbias,
		       curand_state_ptr, handle);
      
      for(int step = 1; step < const_k; step++) {//Repeat as necessary for k
      	gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples,
      		  W, hbias, vbias, curand_state_ptr, handle);
      }
    }
  }
  
  __global__ void write_results_to_memory(DTYPE * data, DTYPE * W, 
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
      // __shared__ DTYPE dhbias_shared[MAX_THREAD_SQUARE_EDGE];
      // __shared__ DTYPE dvbias_shared[MAX_THREAD_SQUARE_EDGE];
      int shared_i = i % MAX_THREAD_SQUARE_EDGE;
      int shared_j = j % MAX_THREAD_SQUARE_EDGE;
      dW_shared[shared_i][shared_j] = 0;
      // if(j == 0) dhbias_shared[shared_i] = 0;
      // if(i == 0) dvbias_shared[shared_j] = 0;
      for(int batch_i = 0; batch_i < const_batch_size; batch_i++) {
	DTYPE * ph_mean  = &ph_mean_batch   [batch_i * const_n_hidden];
	DTYPE * nh_means = &nh_means_batch  [batch_i * const_n_hidden];
	DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
	DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
	dW_shared[shared_i][shared_j] += ph_mean[i] * input[j] - nh_means[i] * nv_samples[j];
	if(j == 0) dhbias[i] += ph_mean[i] - nh_means[i];
	if(i == 0) dvbias[j] += input[j] - nv_samples[j];
      }
      //Surprisingly enough, this is not a race condition, because
      //each thread only depends on itself for this computation
      DTYPE * W_row_i = &W[const_n_visible * i];
      W_row_i[j] = W_row_i[j] + lr * (dW_shared[shared_i][shared_j] / const_batch_size - wc * W_row_i[j]);
      if(j == 0) hbias[i] += lr * dhbias[i] / const_batch_size;
      if(i == 0) vbias[j] += lr * dvbias[j] / const_batch_size;
    }
  }

  void RBM_dynamic_cublas::contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE * dev_data) {
    reset_d_arrays();
    
    if(batch_size > MAX_THREADS) {
      cerr << "ERROR: batch_size cannot exceed 1024" << endl;
    }
    
    // GET_TIME(k1_t1);
    // cerr << "time: " << k1_t1 << endl;
    int cd_blocks = 1 + (batch_size - 1) / NUM_BATCH_THREADS_PER_BLOCK;
    cd_gpu <<< cd_blocks, NUM_BATCH_THREADS_PER_BLOCK, 0, stream>>>
      (dev_data, curr_i, data_num_cols, dev_W, dev_hbias, dev_vbias,
       dev_curand_states, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, curand_batch_width, dev_handle);
    cudaDeviceSynchronize();
    // GET_TIME(k1_t2);
    // cerr << "k1 time: " << get_duration(k1_t1, k1_t2) << endl;
    // CUDA_CHECK(cudaGetLastError());
    
    DTYPE * array = new DTYPE[n_hidden * batch_size];
    cudaMemcpy(array, dev_ph_sample_batch, sizeof(DTYPE) * n_hidden * batch_size, cudaMemcpyDeviceToHost);
    string filename = "array2.dat";
    saveArray(array, n_hidden * batch_size, filename);
    delete[] array;

    
    // cerr << "Initiating write\n";
    dim3 num_blocks, num_threads;
    dims_to_num_threads_and_blocks(n_visible, n_hidden, num_blocks, num_threads);
    // GET_TIME(k2_t1);
    write_results_to_memory <<< num_blocks, num_threads, 0, stream>>>
      (dev_data, dev_W, lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);
    // cudaDeviceSynchronize();
    // GET_TIME(k2_t2);
    // cerr << "k2 time: " << get_duration(k2_t1, k2_t2) << endl;
    // CUDA_CHECK(cudaGetLastError());
  }
  
  //Can the need for this function be removed by using that weird
  //type checking mechanism I saw in that CUDA sample code?
  void RBM_dynamic_cublas::allocate_special_memory() {
    // data = new DTYPE[data_num_rows * data_num_cols];
    // for(int i = 0; i < data_num_rows * data_num_cols; i++) {
    //   data[i] = (DTYPE) int_data[i];
    // }
    
    CUDA_CHECK(cudaMalloc((void**)&dev_ph_sample_batch , sizeof(DTYPE) * n_hidden  * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nv_samples_batch, sizeof(DTYPE) * n_visible * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nh_samples_batch, sizeof(DTYPE) * n_hidden  * batch_size));
    
    CUDA_CHECK(cudaMalloc((void**)&dev_W, n_hidden * n_visible * sizeof(DTYPE)));
    matrixToArray (W, WArray, n_hidden, n_visible);
    
    CUDA_CHECK(cudaMemcpy(dev_W, WArray, n_hidden * n_visible * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));
    
    // CUDA_CHECK(cudaMalloc((void**)&dev_data,
    // 			  data_num_rows * data_num_cols * sizeof(DTYPE)));
    // CUDA_CHECK(cudaMemcpy(dev_data, data, data_num_rows * data_num_cols * sizeof(DTYPE),
    // 			  cudaMemcpyHostToDevice));
  }
    void RBM_dynamic_cublas::reset_d_arrays() {
    //Since dW_pitch is the width of the dev_dW array rows, we
    //multiply by the number of rows (n_hidden) to get the number of
    //bytes to reset:
    CUDA_CHECK(cudaMemset(dev_dhbias, 0, n_hidden  * sizeof(DTYPE)));
    CUDA_CHECK(cudaMemset(dev_dvbias, 0, n_visible * sizeof(DTYPE)));
    
    // CUDA_CHECK(cudaMemset(dev_ph_mean_batch   , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nv_means_batch  , 0, sizeof(DTYPE) * n_visible * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nh_means_batch  , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    // CUDA_CHECK(cudaMemset(dev_ph_sample_batch , 0, sizeof(DTYPE) * n_hidden  * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nv_samples_batch, 0, sizeof(DTYPE) * n_visible * batch_size));
    // CUDA_CHECK(cudaMemset(dev_nh_samples_batch, 0, sizeof(DTYPE) * n_hidden  * batch_size));
  }
  
  void RBM_dynamic_cublas::copy_matrices_to_host() {
    CUDA_CHECK(cudaMemcpy(WArray, dev_W, n_hidden * n_visible * sizeof(DTYPE), 
			  cudaMemcpyDeviceToHost));
    arrayToMatrix(WArray, W, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(vbias, dev_vbias, n_visible * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbias, dev_hbias, n_hidden  * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
  }

    
  // void RBM_dynamic_cublas::saveWeightMatrix() {
  //   cout << "RBM_dynamic_cublas saveWeightMatrix" << endl;
  //   copy_matrices_to_host();
  //   matrixToArray (W, WArray, n_hidden, n_visible);
  //   string wFilename(MATRIX_FILENAME);
  //   saveMatrix(WArray, (size_t) n_hidden, (size_t) n_visible, wFilename);
  //   string hbiasFilename("hbias.dat");
  //   saveArray(hbias, (size_t) n_hidden, hbiasFilename);
  //   string vbiasFilename("vbias.dat");
  //   saveArray(vbias, (size_t) n_visible, vbiasFilename);
  // }

  RBM_dynamic_cublas::RBM_dynamic_cublas(int size, int n_v, int n_h, int b_size, int k,
					 DTYPE **w, DTYPE *hb, DTYPE *vb, 
					 int data_num_rows, int data_num_cols) : baseline::RBM(size, n_v, n_h, b_size, k, w, hb, vb, data_num_rows, data_num_cols) {
    cudaMalloc((void**) &dev_handle, sizeof(cublasHandle_t));
    init_cublas_handle<<<1,1>>>(dev_handle);
    // if(stream != NULL) {
    //   init_cublas_handle_stream<<<1,1>>>(dev_handle, *stream);
    // }
  }
  
  
  RBM_dynamic_cublas::~RBM_dynamic_cublas() {
    destroy_cublas_handle<<<1,1>>>(dev_handle);
    cudaFree(dev_handle);
  }
}
