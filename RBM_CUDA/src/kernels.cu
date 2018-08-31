/*
This file contains all device functions needed by CUDA. (some other cu files may override these/ use different ones)
 */

#include "../include/kernels.h"
#include <curand_kernel.h>
#include "../include/cuda_utils.h"
#include "../include/constants.h"

#define DYNAMIC_PARALLELISM

/*This function propagates the visible units activation upwards to the hidden units*/
__device__ DTYPE propup(DTYPE *v, DTYPE *w, DTYPE b) {
  DTYPE pre_sigmoid_activation = 0.0;
  //This loop will be quite slow because 1 thread is making 2 *
  //const_n_visible individual global memory accesses.
  for(int j=0; j<const_n_visible; j++) {
    pre_sigmoid_activation += w[j] * v[j];
  }
  pre_sigmoid_activation += b;
  DTYPE pre_sigmoid_value = 1.0 / (1.0 + exp(-pre_sigmoid_activation));
  return(pre_sigmoid_value); //sigmoid(pre_sigmoid_activation);
}

//This still needs to be serial, since pre_sigmoid_activation would
//need to be located in global memory. TODO: Upgrade to parallel.
/*This function propagates the hidden units activation downwards to the visible units*/
__device__ DTYPE propdown(DTYPE *h, int j, DTYPE b, DTYPE * W, size_t W_pitch) {
  DTYPE pre_sigmoid_activation = 0.0;
  for(int i = 0; i < const_n_hidden; i++) {
    DTYPE * W_i = get_row_pitch_ptr(W, W_pitch, i);
    pre_sigmoid_activation += W_i[j] * h[i];
  }
  pre_sigmoid_activation += b;
  return 1.0 / (1.0 + exp(-pre_sigmoid_activation)); //sigmoid(pre_sigmoid_activation);
}

__global__ void sample_h_given_v_gpu_kernel(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample,
					    DTYPE * W, size_t W_pitch, DTYPE * hbias, 
					    int last_k, int * tot_hones_temp, int * tot_hones,
					    curandState_t * curand_state_ptr) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < const_n_hidden) {
    DTYPE * W_row_i = get_row_pitch_ptr(W, W_pitch, i);
    mean[i] = propup(v0_sample, W_row_i, hbias[i]);//This can be optimized with the delta product
    if(last_k == 0) {
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[i]);
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[i]);
      #endif
      sample[i] = r < mean[i];
      if(sample[i] == 1) {
      	atomicAdd(tot_hones_temp, 1);
      	atomicAdd(tot_hones     , 1);
      }
    }
  }
}

/*
//PRE: The block size for this kernel should be
//     SUB_KERNEL_NUM_THREADS. 
__global__ void sample_h_given_v_gpu_kernel_proto(int *v0_sample, DTYPE *mean, int *sample,
						  DTYPE * W, size_t W_pitch, DTYPE * hbias, 
						  int last_k, int * tot_hones_temp, int * tot_hones,
						  curandState_t * curand_state_ptr) {
  //Theory: v0_sample and W_row_i are sent to propup and each idx is
  //read individually, and this causes some slow down.
  //Would it be faster to copy those (in parallel) over to shared
  //memory, then copy back? 
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if((i < const_n_hidden) && (j < const_n_visible)) {
    DTYPE * W_row_i = get_row_pitch_ptr(W, W_pitch, i);
    // mean[i] = propup(v0_sample, W_row_i, hbias[i], test_value);
      
    // DTYPE pre_sigmoid_activation = 0.0;
    // for(int j=0; j<const_n_visible; j++) {
    __shared__ DTYPE psa_shared[MAX_THREAD_SQUARE_EDGE];
    if(j % MAX_THREAD_SQUARE_EDGE == 0) {
      psa_shared[i % MAX_THREAD_SQUARE_EDGE] = 0;//Confusing, but correct...
    }
    atomicAdd(&psa_shared[i % MAX_THREAD_SQUARE_EDGE], W_row_i[j] * v0_sample[j]);
    __syncthreads();
      
    if(j % MAX_THREAD_SQUARE_EDGE == 0) {
      atomicAdd(&mean[i], psa_shared[i % MAX_THREAD_SQUARE_EDGE]);
    }
    // pre_sigmoid_activation += W_row_i[j] * v0_sample[j];
    // }
    // pre_sigmoid_activation += hbias[i];
    if(j == 0) {
      // mean[i] = 1.0 / (1.0 + exp(-pre_sigmoid_activation));
      mean[i] += hbias[i];
      mean[i] = 1.0 / (1.0 + exp(-mean[i]));
      if(last_k == 0) {
	DTYPE r = curand_uniform_DTYPE(&curand_state_ptr[i]);
	sample[i] = r < mean[i];
	if(sample[i] == 1) {
	  atomicAdd(tot_hones_temp, 1);
	  atomicAdd(tot_hones     , 1);
	}
      }
    }
  }
}
*/

//Function that sets up the kernel call.
__device__ void sample_h_given_v_gpu(DTYPE *v0_sample, DTYPE *mean, DTYPE *sample,
				     DTYPE * W, size_t W_pitch, DTYPE * hbias, 
				     int last_k, int * tot_hones_temp, int * tot_hones,
				     curandState_t * curand_state_ptr) {
#ifdef PROTO_KERNEL
  dim3 num_blocks, num_threads;
  dims_to_num_threads_and_blocks_gpu(const_n_visible, const_n_hidden,
				     num_blocks, num_threads);
  sample_h_given_v_gpu_kernel_proto<<<num_blocks, num_threads>>>
    (v0_sample, mean, sample, W, W_pitch, hbias, last_k,
     tot_hones_temp, tot_hones, curand_state_ptr);
#else
  #ifdef DYNAMIC_PARALLELISM
  int num_blocks_h = (const_n_hidden  / SUB_KERNEL_NUM_THREADS) + 1;
  sample_h_given_v_gpu_kernel<<<num_blocks_h, SUB_KERNEL_NUM_THREADS>>>
    (v0_sample, mean, sample, W, W_pitch, hbias, last_k,
     tot_hones_temp, tot_hones, curand_state_ptr);
  #else
  for(int i = 0; i < const_n_hidden; i++) {
    DTYPE * W_row_i = get_row_pitch_ptr(W, W_pitch, i);
    mean[i] = propup(v0_sample, W_row_i, hbias[i]);
    if(last_k == 0) {
      //Just access 0 since dyn parallel isn't being used:
      #ifdef USING_DOUBLES
      DTYPE r = curand_uniform_double(&curand_state_ptr[0]);
      #else
      DTYPE r = curand_uniform(&curand_state_ptr[0]);
      #endif
      sample[i] = r < mean[i];
      if(sample[i] == 1) {
      	atomicAdd(tot_hones_temp, 1);
      	atomicAdd(tot_hones     , 1);
      }
    }
  }
  #endif
#endif
  cudaDeviceSynchronize();
}

__global__ void sample_v_given_h_gpu_kernel(DTYPE *h0_sample, DTYPE *mean,
					    DTYPE *sample, DTYPE * W, size_t W_pitch,
					    DTYPE * vbias, 
					    int * tot_vones_temp, int * tot_vones,
					    curandState_t * curand_state_ptr) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if(j < const_n_visible) {
    DTYPE r;
    #ifdef USING_DOUBLES
    r = curand_uniform_double(&curand_state_ptr[j]);
    #else
    r = curand_uniform(&curand_state_ptr[j]);
    #endif
    mean[j] = propdown(h0_sample, j, vbias[j], W, W_pitch);
    sample[j] = r < mean[j];
    
    /* this part is for developing purpose*/
    if(sample[j] == 1) {
      atomicAdd(tot_vones_temp, 1);
      atomicAdd(tot_vones, 1);
    }
  }
}

__device__ void sample_v_given_h_gpu(DTYPE *h0_sample, DTYPE *means, DTYPE *samples,
				     DTYPE * W, size_t W_pitch, DTYPE * vbias, 
				     int * tot_vones_temp, int * tot_vones,
				     curandState_t * curand_state_ptr) {
  #ifdef DYNAMIC_PARALLELISM
  int num_blocks_v = (const_n_visible / SUB_KERNEL_NUM_THREADS) + 1;
  sample_v_given_h_gpu_kernel<<<num_blocks_v, SUB_KERNEL_NUM_THREADS>>>
    (h0_sample, means, samples, W, W_pitch, vbias,
     tot_vones_temp, tot_vones, curand_state_ptr);
  cudaDeviceSynchronize();
  #else
  __device__ DTYPE propdown(DTYPE *h, int j, DTYPE b, DTYPE * W, size_t W_pitch) {
    DTYPE pre_sigmoid_activation = 0.0;
    for(int i = 0; i < const_n_hidden; i++) {
      DTYPE * W_i = get_row_pitch_ptr(W, W_pitch, i);
      pre_sigmoid_activation += W_i[j] * h[i];
    }
  pre_sigmoid_activation += b;
  return 1.0 / (1.0 + exp(-pre_sigmoid_activation)); //sigmoid(pre_sigmoid_activation);
  }
  ///////
  DTYPE r;
  for(int j = 0; j < const_n_visible; j++) {
    //just access 0 since dyn parallel isn't being used:
    #ifdef USING_DOUBLES
    r = curand_uniform_double(&curand_state_ptr[0]);
    #else
    r = curand_uniform(&curand_state_ptr[0]);
    #endif
    means[j] = propdown(h0_sample, j, vbias[j], W, W_pitch);
    samples[j] = r < means[j];
    
    /* this part is for developing purpose*/
    if(samples[j] == 1) {
      atomicAdd(tot_vones_temp, 1);
      atomicAdd(tot_vones, 1);
    }
  }
  #endif
}

  
/*This function implements one step of Gibbs sampling, starting from the hidden state*/
__device__ void gibbs_hvh(DTYPE *h0_sample, DTYPE *nv_means,
			  DTYPE *nv_samples, DTYPE *nh_means,
			  DTYPE *nh_samples, DTYPE * W, int W_pitch,
			  DTYPE * hbias, DTYPE * vbias, int last_k,
			  int * tot_hones_temp, int * tot_hones,
			  int * tot_vones_temp, int * tot_vones,
			  curandState_t * curand_state_ptr) {
  sample_v_given_h_gpu(h0_sample, nv_means, nv_samples, W, W_pitch, vbias,
		       tot_vones_temp, tot_vones, curand_state_ptr);
  sample_h_given_v_gpu(nv_samples, nh_means, nh_samples, W, W_pitch, hbias, last_k,
		       tot_hones_temp, tot_hones, curand_state_ptr);
}

__global__ void cd_gpu(DTYPE * data, int curr_i, 
		       int data_num_cols, int * tot_vones_temp, int * tot_hones_temp,
		       int * tot_vones, int * tot_hones, DTYPE * W, size_t W_pitch,
		       DTYPE * hbias, DTYPE * vbias,
		       curandState_t * curand_states,
		       DTYPE * ph_mean_batch, DTYPE * nv_means_batch,
		       DTYPE * nh_means_batch, DTYPE * ph_sample_batch,
		       DTYPE * nv_samples_batch, DTYPE * nh_samples_batch,
		       int curand_batch_width) {
  int batch_i = blockDim.x * blockIdx.x + threadIdx.x;
  if(batch_i < const_batch_size) {
    DTYPE * ph_mean  = &ph_mean_batch[batch_i * const_n_hidden];
    DTYPE * nv_means = &nv_means_batch[batch_i * const_n_visible];
    DTYPE * nh_means = &nh_means_batch[batch_i * const_n_hidden];
    DTYPE * ph_sample   = &ph_sample_batch[batch_i * const_n_hidden];
    DTYPE * nv_samples  = &nv_samples_batch[batch_i * const_n_visible];
    DTYPE * nh_samples  = &nh_samples_batch[batch_i * const_n_hidden];
    
    curandState_t * curand_state_ptr = &curand_states[batch_i * curand_batch_width];
    DTYPE * input = &data[data_num_cols * (curr_i*const_batch_size+batch_i)];
    int last_k = 0;
    sample_h_given_v_gpu(input, ph_mean, ph_sample, W, W_pitch, hbias, last_k,
			 tot_hones_temp, tot_hones, curand_state_ptr);
    // step 0 to k-1
    for(int step = 0; step < const_k - 1; step++) {
      if(step == 0) {
	gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples,
		  W, W_pitch, hbias, vbias, last_k,
		  tot_hones_temp, tot_hones, tot_vones_temp, tot_vones,
		  curand_state_ptr);//1
      } else {
	gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples,
		  W, W_pitch, hbias, vbias, last_k,
		  tot_hones_temp, tot_hones, tot_vones_temp, tot_vones,
		  curand_state_ptr);//2
      }
    }
    /* step k */
    last_k = 1;
    if(const_k == 1) {
      gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples,
		W, W_pitch, hbias, vbias, last_k,
		tot_hones_temp, tot_hones, tot_vones_temp, tot_vones,
		curand_state_ptr);//3
    } else {
      gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples,
		W, W_pitch, hbias, vbias, last_k,
		tot_hones_temp, tot_hones, tot_vones_temp, tot_vones,
		curand_state_ptr);//4
    }
    //k = 1: 3 (last_k = 1)
    //k = 2: 1 (last_k = 0), 4 (last_k = 1)
  }
}


//Putting dW in shared memory for this function made it many times
//faster, but dhbias and dvbias in shared memory makes it slightly
//slower, probably because of the if stmts that are necessary. 
__global__ void write_results_to_memory(DTYPE * data, DTYPE * W, int W_pitch,
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
    DTYPE * W_row_i = get_row_pitch_ptr(W, W_pitch, i);
    W_row_i[j] = W_row_i[j] + lr * (dW_shared[shared_i][shared_j] / const_batch_size - wc * W_row_i[j]);
    if(j == 0) hbias[i] += lr * dhbias[i] / const_batch_size;
    if(i == 0) vbias[j] += lr * dvbias[j] / const_batch_size;
  }
}
