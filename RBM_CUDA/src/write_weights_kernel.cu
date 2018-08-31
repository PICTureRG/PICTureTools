#include "../include/write_weights_kernel.h"
#include "../include/constants.h"

__global__
void write_weights(DTYPE * data, DTYPE * W,
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
    
    // int shared_i = i % MAX_THREAD_SQUARE_EDGE;
    // int shared_j = j % MAX_THREAD_SQUARE_EDGE;
    dW_shared[threadIdx.y][threadIdx.x] = 0;
    //Run the computations for each batch:
    for(int batch_i = 0; batch_i < const_batch_size; batch_i++) {
      DTYPE * ph_mean  = &ph_mean_batch   [batch_i * const_n_hidden];
      DTYPE * nh_means = &nh_means_batch  [batch_i * const_n_hidden];
      DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
      DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
      dW_shared[threadIdx.y][threadIdx.x] += ph_mean[i] * input[j] - nh_means[i] * nv_samples[j];
    }
    //Surprisingly enough, this is not a race condition, because
    //each thread only depends on itself for this computation
    #ifdef WEIGHT_MATRIX_PADDING
    DTYPE * W_row_i = &W[(const_pitch/sizeof(DTYPE)) * i];
    #else
    DTYPE * W_row_i = &W[const_n_visible * i];
    #endif
    
    DTYPE new_ij_value = W_row_i[j] + lr * (dW_shared[threadIdx.y][threadIdx.x] / const_batch_size - wc * W_row_i[j]);
    W_row_i[j] = new_ij_value;
#ifdef MULTI_WEIGHT_MATRIX
#ifndef EFFICIENT_TRANSPOSE
    //Just write W to W2 in transpose:
    #ifdef WEIGHT_MATRIX_PADDING
    DTYPE * W2_row_j=&W2[(const_pitch2/sizeof(DTYPE)) * j];
    W2_row_j[i] = new_ij_value;
    // W2[const_pitch2 * j + i] = new_ij_value;
    #else
    DTYPE * W2_row_j=&W2[const_n_hidden * j];
    W2_row_j[i] = new_ij_value;
    // W2[const_n_hidden * j + i] = new_ij_value;
    #endif
#endif
#endif
  }
}
//dp_matrix (float) 20 20 = 19.524
// W is: 
// -0.0435138 -0.0457548 -0.0459917 -0.0442509 -0.0450698 
// -0.0340619 -0.035235 -0.0345938 -0.0357267 -0.0345877 
// -0.031528 -0.0297256 -0.0295831 -0.0318596 -0.0299962 
// -0.0332557 -0.03282 -0.0333473 -0.0348455 -0.0342421 
// -0.0481603 -0.0495691 -0.0494114 -0.050075 -0.0486808 
// ...
// hbias is: 
// 0.178762
// 0.158194
// 0.154333
// 0.139903
// 0.160019
// ...
// vbias is: 
// -0.108701
// -0.111801
// -0.111901
// -0.110601
// -0.111101
// ...

