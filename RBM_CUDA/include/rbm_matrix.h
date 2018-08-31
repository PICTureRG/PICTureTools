#ifndef INCLUDED_rbm_matrix
#define INCLUDED_rbm_matrix

#include "../include/rbm_baseline.h"
#include "rbm_baseline.h"
#include "cublas_v2.h"

/* #define MULTI_WEIGHT_MATRIX */
//Leave off, MULTI_WEIGHT_MATRIX capability doesn't affect performance. 

namespace matrix {
  class RBM_matrix;
  /* __device__ void matrix_dot_vector(DTYPE * matrix, int m, int n, */
  /* 				    DTYPE * vector, DTYPE * result, bool transpose); */
  /* __global__ void finish_sampling_kernel(DTYPE * mean, DTYPE * sample, DTYPE * bias, */
  /* 					 int length, curandState_t * curand_state_ptr); */
  /* __device__ void sample_h_given_v(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample, */
  /* 					   DTYPE * W, DTYPE * hbias, int last_k, */
  /* 					   curandState_t * curand_state_ptr); */
  /* __device__ void sample_h_given_v(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample, */
  /* 					   DTYPE * W, DTYPE * hbias, int last_k, */
  /* 					   curandState_t * curand_state_ptr); */
  /* __device__ void gibbs_hvh(DTYPE *h0_sample, DTYPE *nv_means, */
  /* 			    DTYPE *nv_samples, DTYPE *nh_means, */
  /* 			    DTYPE *nh_samples, DTYPE * W, */
  /* 			    DTYPE * hbias, DTYPE * vbias, int last_k, */
  /* 			    curandState_t * curand_state_ptr); */

}

class matrix::RBM_matrix : public baseline::RBM {
 public:

#ifdef MULTI_WEIGHT_MATRIX
  DTYPE * WArray2;
  DTYPE * dev_W2;
#endif


  void sample_v_given_h(DTYPE *dev_h0_sample, DTYPE *dev_mean, DTYPE *dev_sample);
  void sample_h_given_v(DTYPE *dev_v0_sample, DTYPE *dev_mean, DTYPE *dev_sample);
  void gibbs_hvh(DTYPE *h0_sample, DTYPE *nv_means, DTYPE *nv_samples,
		 DTYPE *nh_means, DTYPE *nh_samples);

  //"Override" the int versions of these:
  DTYPE *dev_ph_sample_batch;
  DTYPE *dev_nv_samples_batch;
  DTYPE *dev_nh_samples_batch;
  DTYPE *dev_data;
  DTYPE *data;

  cublasHandle_t handle;
  
  RBM_matrix(int, int, int, int, int, DTYPE**, DTYPE*, DTYPE*, int, int);
  ~RBM_matrix();
  
  void reset_d_arrays();

  void contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE *);
  void allocate_special_memory();
  void copy_matrices_to_host();
  /* void saveWeightMatrix(); */

  /* void printParams(); */

};

#endif
