#ifndef INCLUDED_rbm_delta_product
#define INCLUDED_rbm_delta_product

#include <vector>
/* #include <curand_kernel.h> */
/* #include <boost/date_time/posix_time/posix_time.hpp> */
#include "rbm_baseline.h"
#include "cublas_v2.h"

using namespace std;

#define MULTI_WEIGHT_MATRIX

namespace delta_product {
  class RBM_delta;
  /* void test_rbm(int, int, DTYPE, int, int, const char*, int, int, DTYPE); */
}

class delta_product::RBM_delta : public baseline::RBM {

 public:
  #ifdef MULTI_WEIGHT_MATRIX
  DTYPE * WArray2;
  DTYPE * dev_W2;
  #endif
  
  //------Copied from rbm_dynamic_cublas.h
  DTYPE *dev_ph_sample_batch;
  DTYPE *dev_nv_samples_batch;
  DTYPE *dev_nh_samples_batch;
  DTYPE *dev_data;
  DTYPE *data;
  cublasHandle_t * dev_handle;
  cublasHandle_t * handle;
  //------
  
  int * dev_v_diffs;
  int * dev_h_diffs;
  //NOTE: The dot product calculations are all performed like a
  //matrix.vector operation, but then a bias is added and a sigmoid
  //operation is performed in order to get the means. When performing
  //the delta product calculations using the dev_v_diffs and
  //dev_h_diffs arrays, we need the original matrix.vector result, and
  //these arrays store that information for each hidden/visible phase: 
  DTYPE * dev_h_dot_product_batch;
  DTYPE * dev_v_dot_product_batch;
  
  RBM_delta(int, int, int, int, int, DTYPE**,
	    DTYPE*, DTYPE*, int, int);
  ~RBM_delta();
  void reset_d_arrays();
  void contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE *);
  void allocate_special_memory();
  void copy_matrices_to_host();
  /* void saveWeightMatrix(); */

  /* void contrastive_divergence(int, DTYPE, DTYPE); */
  /* void copy_matrices_to_host(); */
 
  /* void sample_h_given_v(int*, DTYPE*, int*); */
  /* void sample_v_given_h(int*, DTYPE*, int*); */
  /* DTYPE propup(int*, DTYPE*, DTYPE); */
  /* DTYPE propdown(int*, int, DTYPE); */
  /* void gibbs_hvh(int*, DTYPE*, int*, DTYPE*, int*); */
  
  /* void reconstruct(int*, DTYPE*); */
  /* void printParams(); */
  /* void printExpResult(DTYPE, DTYPE); */
};

#endif
