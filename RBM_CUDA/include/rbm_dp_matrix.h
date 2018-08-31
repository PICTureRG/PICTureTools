#ifndef INCLUDED_rbm_dp_matrix
#define INCLUDED_rbm_dp_matrix

#include <vector>
/* #include <curand_kernel.h> */
/* #include <boost/date_time/posix_time/posix_time.hpp> */
#include "rbm_baseline.h"
#include "cublas_v2.h"
#include "utils.h"
#include "constants.h"

using namespace std;


/* With MULTI_WEIGHT_MATRIX on k = 15, e = 15, bs = 100: 17.9011 */
/* Without: 29.2165 */

namespace dp_matrix {
  class RBM_dp_matrix;
  /* void test_rbm(int, int, DTYPE, int, int, const char*, int, int, DTYPE); */
}

class dp_matrix::RBM_dp_matrix : public baseline::RBM {

 public:
  #ifdef MULTI_WEIGHT_MATRIX
  DTYPE * WArray2;
  DTYPE * dev_W2;
  #endif

  #ifdef WEIGHT_MATRIX_PADDING
  size_t pitch;
  #ifdef MULTI_WEIGHT_MATRIX
  size_t pitch2;//for transpose
  #endif
  DTYPE * dev_W_cublas;
  #endif

  #ifndef BIT_CODING
  bool * vdirections;
  bool * hdirections;
  #endif
  
  cublasHandle_t host_handle;
  
  //------Copied from rbm_dynamic_cublas.h
  DTYPE *dev_ph_sample_batch;
  DTYPE *dev_nv_samples_batch;
  DTYPE *dev_nh_samples_batch;
  DTYPE *dev_data;
  DTYPE *data;
  /* cublasHandle_t * dev_handle; */
  /* cublasHandle_t * handle; */
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

  void save_sample_changes(DTYPE * mean, DTYPE * sample,
			   DTYPE * prev_sample, int * diffs, int length
			   #ifndef BIT_CODING
			   , bool * direction
			   #endif
			   );
  void add_bias(DTYPE * mean, DTYPE * dot_product,
		DTYPE * bias, int length);

  void sample_v_given_h_matrix(DTYPE *dev_h0_sample, DTYPE *dev_v_mean,
			       DTYPE *v_sample, DTYPE *prev_v_sample);

  void dpmm(int m, int n, int * diffs,
	    DTYPE * dot_product, bool transpose
	    #ifndef BIT_CODING
	    , bool * directions
	    #endif
	    );

  void sample_h_given_v_delta(DTYPE * v0_sample, DTYPE * h_mean,
			      DTYPE * h_sample, DTYPE * prev_h_sample);

  void sample_v_given_h_delta(DTYPE * h0_sample, DTYPE * v_mean,
			      DTYPE * v_sample, DTYPE * prev_v_sample);

  void gibbs_hvh_delta(DTYPE *h0_sample, DTYPE *nv_means, DTYPE *nv_samples,
		       DTYPE *nh_means, DTYPE *nh_samples);
    

    
  
  RBM_dp_matrix(int, int, int, int, int, DTYPE**,
	    DTYPE*, DTYPE*, int, int);
  ~RBM_dp_matrix();
  void reset_d_arrays();
  void contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE *);
  void allocate_special_memory();
  void copy_matrices_to_host();
  /* void saveWeightMatrix(); */

  void sample_h_given_v_matrix(DTYPE *dev_v0_sample, DTYPE *dev_mean, DTYPE *dev_sample);
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
