#ifndef INCLUDED_kernels
#define INCLUDED_kernels
#include <curand_kernel.h>
#include "constants.h"

__device__ DTYPE propup(DTYPE *v, DTYPE *w, DTYPE b);

__device__ DTYPE propdown(DTYPE *h, int j, DTYPE b, DTYPE * W, size_t W_pitch);

__global__ void sample_h_given_v_gpu_kernel(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample,
					    DTYPE * W, size_t W_pitch, DTYPE * hbias, 
					    int last_k, int * tot_hones_temp,
					    int * tot_hones,
					    curandState_t * curand_state_ptr);

__global__ void sample_h_given_v_gpu_kernel_proto(DTYPE *v0_sample, DTYPE *mean,
						  DTYPE *sample, DTYPE * W, size_t W_pitch,
						  DTYPE * hbias, int last_k,
						  int * tot_hones_temp, int * tot_hones,
						  curandState_t * curand_state_ptr);

__device__ void sample_h_given_v_gpu(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample,
				     DTYPE * W, size_t W_pitch, DTYPE * hbias, 
				     int last_k, int * tot_hones_temp, int * tot_hones,
				     curandState_t * curand_state_ptr);

__global__ void sample_v_given_h_gpu_kernel(DTYPE *h0_sample, DTYPE *mean,
					    DTYPE *sample, DTYPE * W, size_t W_pitch,
					    DTYPE * vbias, 
					    int * tot_vones_temp, int * tot_vones,
					    curandState_t * curand_state_ptr);
__device__ void sample_v_given_h_gpu(DTYPE *h0_sample, DTYPE *mean,
				     DTYPE *sample, DTYPE * W, size_t W_pitch,
				     DTYPE * vbias, 
				     int * tot_vones_temp, int * tot_vones,
				     curandState_t * curand_state_ptr);

__device__ void gibbs_hvh(DTYPE *h0_sample, DTYPE *nv_means,
			  DTYPE *nv_samples, DTYPE *nh_means,
			  DTYPE *nh_samples, DTYPE * W, int W_pitch,
			  DTYPE * hbias, DTYPE * vbias, int last_k,
			  int * tot_hones_temp, int * tot_hones,
			  int * tot_vones_temp, int * tot_vones,
			  curandState_t * curand_state_ptr);

__global__ void cd_gpu(DTYPE * data, int curr_i, 
		       int data_num_cols, int * tot_vones_temp, int * tot_hones_temp,
		       int * tot_vones, int * tot_hones, DTYPE * W, size_t W_pitch,
		       DTYPE * hbias, DTYPE * vbias,
		       curandState_t * curand_states,
		       DTYPE * ph_mean_batch, DTYPE * nv_means_batch,
		       DTYPE * nh_means_batch, DTYPE * ph_sample_batch,
		       DTYPE * nv_samples_batch, DTYPE * nh_samples_batch,
		       int curand_batch_width);

__global__ void write_results_to_memory(DTYPE * data, DTYPE * W, int W_pitch,
					DTYPE lr, DTYPE wc, DTYPE * ph_mean_batch,
					DTYPE * nv_means_batch, DTYPE * nh_means_batch,
					DTYPE * ph_sample_batch, DTYPE * nv_samples_batch,
					DTYPE * nh_samples_batch, DTYPE * hbias,
					DTYPE * vbias, DTYPE * dhbias, DTYPE * dvbias,
					int data_num_rows, int data_num_cols,
					int curr_i);

#endif
