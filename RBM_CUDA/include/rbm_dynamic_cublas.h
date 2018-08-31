#ifndef INCLUDED_rbm_dynamic_cublas
#define INCLUDED_rbm_dynamic_cublas

#include "../include/rbm_baseline.h"
#include "rbm_baseline.h"
#include "cublas_v2.h"

namespace dynamic_cublas {
  class RBM_dynamic_cublas;
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

class dynamic_cublas::RBM_dynamic_cublas : public baseline::RBM {
 public:

  //"Override" the int versions of these:
  DTYPE *dev_ph_sample_batch;
  DTYPE *dev_nv_samples_batch;
  DTYPE *dev_nh_samples_batch;
  DTYPE *dev_data;
  DTYPE *data;

  cublasHandle_t * dev_handle;
  
  RBM_dynamic_cublas(int, int, int, int, int, DTYPE**, DTYPE*, DTYPE*, int, int);
  ~RBM_dynamic_cublas();
  
  void reset_d_arrays();

  void contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE*);
  void allocate_special_memory();
  void copy_matrices_to_host();
  /* void saveWeightMatrix(); */

  /* void printParams(); */

};

#endif
