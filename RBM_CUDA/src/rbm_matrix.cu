#include <iostream>
#include "../include/rbm_matrix.h"
#include "../include/rbm_baseline.h"
#include "../include/utils.h"
#include "cublas_v2.h"
#include "../include/constants.h"

using namespace std;

#define SAMPLING_KERNEL_BLOCK_SIZE 32
// #define DEBUG
namespace matrix {

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
    //Need B.A in order to get transposed result in C.
    #ifdef USING_DOUBLES
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

    if(status != CUBLAS_STATUS_SUCCESS) {
      cerr << "gemm error\n";
    }
  }

  __global__ void finish_sampling_kernel(DTYPE * mean, DTYPE * sample, DTYPE * bias,
					 int length, curandState_t * curand_states) {
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if(i < length) {
    //   int batch_i = blockDim.y * blockIdx.y + threadIdx.y;
    //   if(batch_i < const_batch_size) {
    // 	int batch_shift = batch_i * length;
    // 	DTYPE mean_i = 1.0 / (1.0 + exp(-(mean[i + batch_shift] + bias[i])));//TODO: optimize?
    // 	mean[i + batch_shift] = mean_i;
    // 	DTYPE r = curand_uniform_DTYPE(&curand_states[i + batch_shift]);
    // 	sample[i + batch_shift] = r < mean_i;
    //   }
    // }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length * const_batch_size) {
      int hidden_idx = i % length;
      DTYPE mean_i = 1.0 / (1.0 + exp(-(mean[i] + bias[hidden_idx])));
      mean[i] = mean_i;
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_states[i]);
      #else
      DTYPE r = curand_uniform(&curand_states[i]);
      #endif
      sample[i] = r < mean_i;
      // sample[i] = mean_i;
    }
  }

  
  //Average time: 0.000766
  void RBM_matrix::sample_h_given_v(DTYPE *dev_v0_sample, DTYPE *dev_mean, DTYPE *dev_sample) {
    //Goal is to compute (W . v0_sample) + hbias and store in mean,
    //then do the random number thing and store in sample
    // matrix_dot_vector(W, const_n_hidden, const_n_visible, v0_sample, mean, false, handle);
    // cublasDgemv(*handle, transpose ? CUBLAS_OP_N : CUBLAS_OP_T,
    // 	n, m,
    // 	&alpha,
    // 	matrix, n,
    // 	vector, 1,
    // 	&beta,
    // 	result, 1);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
    // 		batch_size, n_hidden, n_visible, 1.0,
    // 		myxsample, n_visible,
    // 		W, n_visible,
    // 		0.0, mean, n_hidden);
    // cudaDeviceSynchronize();
    // GET_TIME(t1);
    
#ifdef MULTI_WEIGHT_MATRIX
    gemm(dev_v0_sample, dev_W2, dev_mean,
	 batch_size, n_hidden, n_visible,
	 false, false, handle);
#else
    gemm(dev_v0_sample, dev_W, dev_mean,
	 batch_size, n_hidden, n_visible,
	 false, true, handle);
#endif
    // cudaDeviceSynchronize();
    // GET_TIME(t2);
    // cerr << "MM time: " << get_duration(t1, t2) << endl;
    
    // cudaDeviceSynchronize();
    // GET_TIME(t3);

    int num_blocks = ((n_hidden * batch_size - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    finish_sampling_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (dev_mean, dev_sample, dev_hbias, n_hidden, dev_curand_states);
    
    // cudaDeviceSynchronize();
    // GET_TIME(t2);
    // cerr << "sample_h_given_v: " << get_duration(t1, t2) << endl;
    // cerr << "finish up time: " << get_duration(t3, t4) << endl;
  }

  //0.001045
  void RBM_matrix::sample_v_given_h(DTYPE *dev_h0_sample, DTYPE *dev_mean, DTYPE *dev_sample) {
    // cudaDeviceSynchronize();
    // GET_TIME(t1);
    
    // matrix_dot_vector(W, const_n_hidden, const_n_visible, h0_sample,
    // 		      mean, true, handle);
    // int num_blocks = ((const_n_visible - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;

#ifdef MULTI_WEIGHT_MATRIX
    gemm(dev_h0_sample, dev_W2, dev_mean,
    	 batch_size, n_visible, n_hidden,//Dims are same AFTER transposition
    	 false, true, handle);
#else
    gemm(dev_h0_sample, dev_W, dev_mean,
    	 batch_size, n_visible, n_hidden,
    	 false, false, handle);
#endif
    int num_blocks = ((n_visible * batch_size - 1) / SAMPLING_KERNEL_BLOCK_SIZE) + 1;
    finish_sampling_kernel<<<num_blocks, SAMPLING_KERNEL_BLOCK_SIZE>>>
      (dev_mean, dev_sample, dev_vbias, n_visible, dev_curand_states);
    
    // cudaDeviceSynchronize();
    // GET_TIME(t2);
    // cerr << "t: " << get_duration(t1, t2) << endl;
  }
  
  void RBM_matrix::gibbs_hvh(DTYPE *h0_sample, DTYPE *nv_means, DTYPE *nv_samples,
			     DTYPE *nh_means, DTYPE *nh_samples) {
    sample_v_given_h(h0_sample, nv_means, nv_samples);
    sample_h_given_v(nv_samples, nh_means, nh_samples);
  }

  
  __global__ void write_results_to_memory(DTYPE * data, DTYPE * W,
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
      // __shared__ DTYPE dhbias_shared[MAX_THREAD_SQUARE_EDGE];
      // __shared__ DTYPE dvbias_shared[MAX_THREAD_SQUARE_EDGE];
      int shared_i = i % MAX_THREAD_SQUARE_EDGE;
      int shared_j = j % MAX_THREAD_SQUARE_EDGE;
      dW_shared[shared_i][shared_j] = 0;
// #ifdef MULTI_WEIGHT_MATRIX
//       dW2_shared[shared_j][shared_i] = 0.0;
// #endif
      // if(j == 0) dhbias_shared[shared_i] = 0;
      // if(i == 0) dvbias_shared[shared_j] = 0;
      for(int batch_i = 0; batch_i < const_batch_size; batch_i++) {
	DTYPE * ph_mean  = &ph_mean_batch   [batch_i * const_n_hidden];
	DTYPE * nh_means = &nh_means_batch  [batch_i * const_n_hidden];
	DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
	DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
	dW_shared[shared_i][shared_j] += ph_mean[i] * input[j] - nh_means[i] * nv_samples[j];
// #ifdef MULTI_WEIGHT_MATRIX
// 	dW2_shared[shared_j][shared_i] += dW_shared[shared_i][shared_j];
// #endif
	if(j == 0) dhbias[i] += ph_mean[i] - nh_means[i];
	if(i == 0) dvbias[j] += input[j] - nv_samples[j];
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

      if(j == 0) hbias[i] += lr * dhbias[i] / const_batch_size;
      if(i == 0) vbias[j] += lr * dvbias[j] / const_batch_size;
    }
  }

  void RBM_matrix::contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE * dev_data) {
    
    // cudaDeviceSynchronize();
    // GET_TIME(k1_t1);
    reset_d_arrays();
    
    if(batch_size > MAX_THREADS) {
      cerr << "ERROR: batch_size cannot exceed 1024" << endl;
    }
    
    DTYPE * dev_input = &dev_data[data_num_cols * (curr_i*batch_size)];
    
    sample_h_given_v(dev_input, dev_ph_mean_batch, dev_ph_sample_batch);
    
    gibbs_hvh(dev_ph_sample_batch, dev_nv_means_batch, dev_nv_samples_batch,
    	      dev_nh_means_batch, dev_nh_samples_batch);
    
    for(int step = 1; step < k; step++) {//Repeat as necessary for k
      gibbs_hvh(dev_nh_samples_batch, dev_nv_means_batch, dev_nv_samples_batch,
    		dev_nh_means_batch, dev_nh_samples_batch);
    }
    // cudaDeviceSynchronize();
    // GET_TIME(k1_t2);
    // cerr << "k1 time: " << get_duration(k1_t1, k1_t2) << endl;
   
    //Save dev_ph_mean_batch
    // DTYPE * array = new DTYPE[n_hidden * batch_size];
    // cudaMemcpy(array, dev_ph_sample_batch, sizeof(DTYPE) * n_hidden * batch_size, cudaMemcpyDeviceToHost);
    // string filename = "array1.dat";
    // saveArray(array, n_hidden * batch_size, filename);
    // delete[] array;
    
    dim3 num_blocks, num_threads;
    dims_to_num_threads_and_blocks(n_visible, n_hidden, num_blocks, num_threads);
    // cudaDeviceSynchronize();
    // GET_TIME(k2_t1);
    //dev_data is correct here for now because the kernel does the necessary translation
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    write_results_to_memory <<< num_blocks, num_threads>>>
      (dev_data, dev_W,
#ifdef MULTI_WEIGHT_MATRIX
       dev_W2,
#endif
       lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
    // GET_TIME(k2_t2);
    // cerr << "k2 time: " << get_duration(k2_t1, k2_t2) << endl;
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
  }
  
  //This the need for this function be removed by using that weird
  //type checking mechanism I saw in that CUDA sample code?
  void RBM_matrix::allocate_special_memory() {
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
  }
  void RBM_matrix::reset_d_arrays() {
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
  
  void RBM_matrix::copy_matrices_to_host() {
    CUDA_CHECK(cudaMemcpy(WArray, dev_W, n_hidden * n_visible * sizeof(DTYPE), 
			  cudaMemcpyDeviceToHost));
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
  
  
  RBM_matrix::RBM_matrix(int size, int n_v, int n_h, int b_size, int k,
			 DTYPE **w, DTYPE *hb, DTYPE *vb, 
			 int data_num_rows, int data_num_cols) : baseline::RBM(size, n_v, n_h, b_size, k, w, hb, vb, data_num_rows, data_num_cols) {
    cublasCreate(&handle);
#ifdef MULTI_WEIGHT_MATRIX
    cout << "Using MULTI_WEIGHT_MATRIX" << endl;
    if(w == NULL) {
      WArray2 = new DTYPE[n_hidden * n_visible];
    }
#endif

    cout << "RBM_matrix constructor\n";
  }
  
  RBM_matrix::~RBM_matrix() {
    cublasDestroy(handle);
    #ifdef SAVE_WEIGHTS
    saveWeightMatrix();
    #endif
#ifdef MULTI_WEIGHT_MATRIX
    delete[] WArray2;
    cudaFree(dev_W2);
#endif
  }
}
