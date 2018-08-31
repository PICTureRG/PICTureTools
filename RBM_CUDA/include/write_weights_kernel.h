#ifndef INCLUDED_write_weights_kernel
#define INCLUDED_write_weights_kernel

#include "rbm_dp_matrix.h"
#include "constants.h"

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
		     int data_num_rows, int data_num_cols, int curr_i);
#endif
