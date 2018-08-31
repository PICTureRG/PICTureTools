/* 
   This file is the implemetation of RBM following the original algorithm.
   It is used as the baseline algorithm for comparison.
   It has been rewritten for CUDA
   
   Notes on code style:
   Ptrs on the host that point to device arrays should be prefixed with "dev_". 
   __constant__ CUDA memory should be prefixed with "const_".

   The CUDA toolkit documentation says:
   "For the highest quality parallel pseudorandom number generation,
   each experiment should be assigned a unique seed. Within an
   experiment, each thread of computation should be assigned a unique
   sequence number. If an experiment spans multiple kernel launches,
   it is recommended that threads between kernel launches be given the
   same seed, and sequence numbers be assigned in a monotonically
   increasing way. If the same configuration of threads is launched,
   random state can be preserved in global memory between launches to
   avoid state setup time. "
   
   With this in mind, this code will allocate a curandState_t object
   for each thread, and init each with a different seed and sequence
   number, where the seed is shifted by the current time. 
   
   Note that the dW array is now completely in shared memory during
   the write_results_to_memory __global__ function. 

   Question: Could I get rid of the nv_means array and just directly
   compute the nv_samples array?
*/


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <string.h>
#include "../include/rbm_baseline.h"
#include "../include/utils.h"
#include "../include/cuda_utils.h"
#include "../include/kernels.h"

#include <curand_kernel.h>
#include <time.h>

// #include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace utils;

namespace baseline {
  
  /* RBM constructor */
  //Note that only one of these RBM classes should be run at a
  //time, since the constant CUDA memory will be overwritten by
  //secondary classes. 
  RBM::RBM(int size, int n_v, int n_h, int b_size, int k,
	   DTYPE **w, DTYPE *hb, DTYPE *vb,
	   int data_num_rows, int data_num_cols) {
    // cerr << "RBM baseline constructor\n";
    cudaStreamCreate(&stream);
    
    N = size;
    n_visible = n_v;
    cudaMemcpyToSymbol(const_n_visible , &n_visible , sizeof(int));
    n_hidden = n_h;
    cudaMemcpyToSymbol(const_n_hidden  , &n_hidden  , sizeof(int));
    batch_size = b_size;
    cudaMemcpyToSymbol(const_batch_size, &batch_size, sizeof(int));
    this->k = k;
    cudaMemcpyToSymbol(const_k         , &k         , sizeof(int));

    this->data_num_rows = data_num_rows;
    this->data_num_cols = data_num_cols;

    //Number of curand objects per batch_i:
    curand_batch_width = n_visible > n_hidden ? n_visible : n_hidden;
    int num_curand_states = batch_size * curand_batch_width;
    cudaMalloc((void**)&dev_curand_states,
	       num_curand_states * sizeof(curandState_t));
    int num_blocks = (num_curand_states / MAX_THREADS) + 1;
#ifdef RANDOM_RUNS
    init_curand<<<num_blocks, MAX_THREADS>>>(dev_curand_states,
					     num_curand_states, time(NULL));
#else
    init_curand<<<num_blocks, MAX_THREADS>>>(dev_curand_states,
					     num_curand_states, 0);
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    //==========================
    //Allocate all device memory
    //==========================


    // allocate_special_memory();
    // CUDA_CHECK(cudaMallocPitch((void**)&dev_W, &W_pitch,
    // 			       n_visible * sizeof(DTYPE), n_hidden));
    
    // cout << "Weight matrix pitch is " << W_pitch << " bytes." << endl;
    // cout << "This implies that " << (W_pitch - (n_visible * sizeof(DTYPE)))
    // 	 << " bytes extra are being used per row." << endl;
    
    // CUDA_CHECK(cudaMalloc((void**)&dev_data, data_num_rows * data_num_cols * sizeof(int)));
    
    CUDA_CHECK(cudaMalloc((void**)&dev_hbias , n_hidden  * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc((void**)&dev_vbias , n_visible * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc((void**)&dev_dhbias, n_hidden  * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc((void**)&dev_dvbias, n_visible * sizeof(DTYPE)));
    
    CUDA_CHECK(cudaMemset(dev_hbias, 0, n_hidden  * sizeof(DTYPE)));
    CUDA_CHECK(cudaMemset(dev_vbias, 0, n_visible * sizeof(DTYPE)));
    
    CUDA_CHECK(cudaMalloc((void**)&dev_tot_vones_temp, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_tot_hones_temp, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_tot_vones     , sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_tot_hones     , sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&dev_ph_mean_batch   , sizeof(DTYPE) * n_hidden  * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nv_means_batch  , sizeof(DTYPE) * n_visible * batch_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_nh_means_batch  , sizeof(DTYPE) * n_hidden  * batch_size));
    // cudaMalloc((void**)&dev_ph_sample_batch , sizeof(int   ) * n_hidden  * batch_size);
    // cudaMalloc((void**)&dev_nv_samples_batch, sizeof(int   ) * n_visible * batch_size);
    // cudaMalloc((void**)&dev_nh_samples_batch, sizeof(int   ) * n_hidden  * batch_size);
    
    last_k = 0;
    /* parameters for debugging and development purpose */
    tot_vones = 0;
    tot_hones = 0;
    tot_vones_temp = 0;
    tot_hones_temp = 0;
    
    /* initializing learning parameters */
    if(w == NULL) {
      WArray = new DTYPE[n_hidden * n_visible];
      
      W = new DTYPE*[n_hidden];
      for(int i=0; i<n_hidden; i++)
	W[i] = new DTYPE[n_visible];
      DTYPE a = 1.0 / n_visible;
      
      for(int i=0; i<n_hidden; i++) {
	for(int j=0; j<n_visible; j++) {
	  W[i][j] = uniform(-a, a);
	}
      }
    } else {
      W = w;
    }
    
    if(hb == NULL) {
      hbias = new DTYPE[n_hidden];
      for(int i=0; i<n_hidden; i++) hbias[i] = 0;
    } else {
      hbias = hb;
    }
    
    if(vb == NULL) {
      vbias = new DTYPE[n_visible];
      for(int i=0; i<n_visible; i++) vbias[i] = 0;
    } else {
      vbias = vb;
    }
    
    /* initialize gradient for updating the parameters when batch_size >= 1 */
    if(batch_size != 0) {
      dhbias = new DTYPE[n_hidden];
      for(int i=0; i<n_hidden; i++) dhbias[i] = 0;

      dvbias = new DTYPE[n_visible];
      for(int i=0; i<n_visible; i++) dvbias[i] = 0;
    }

  }
  
  //This function is for memory that changes in terms of its
  //allocation in child classes. 
  void RBM::allocate_special_memory() {
    this->data = data;
    
    cudaMalloc((void**)&dev_ph_sample_batch , sizeof(int) * n_hidden  * batch_size);
    cudaMalloc((void**)&dev_nv_samples_batch, sizeof(int) * n_visible * batch_size);
    cudaMalloc((void**)&dev_nh_samples_batch, sizeof(int) * n_hidden  * batch_size);
    
    CUDA_CHECK(cudaMallocPitch((void**)&dev_W, &W_pitch,
    			       n_visible * sizeof(DTYPE), n_hidden));
    matrixToArray (W, WArray, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy2D(dev_W, W_pitch, WArray, n_visible * sizeof(DTYPE),
    			    n_visible * sizeof(DTYPE), n_hidden, cudaMemcpyHostToDevice));
  }
  
  /* RBM destructor */
  RBM::~RBM() {
    // saveWeightMatrix();
    if(batch_size != 0) {
      for(int i=0; i<n_hidden; i++) {
	delete[] W[i];
      }
      delete[] dhbias;
      delete[] dvbias;
    }
    else {
      for(int i=0; i<n_hidden; i++) delete[] W[i];
    }
    
    
    delete[] W;
    delete[] hbias;
    delete[] vbias;
    
    delete[] WArray;
    
    cudaStreamDestroy(stream);
    cudaFree(dev_curand_states);
    
    cudaFree(dev_ph_mean_batch);
    cudaFree(dev_ph_sample_batch);
    cudaFree(dev_nv_means_batch);
    cudaFree(dev_nv_samples_batch);
    cudaFree(dev_nh_means_batch);
    cudaFree(dev_nh_samples_batch);
    
    cudaFree(dev_W);
    cudaFree(dev_hbias);
    cudaFree(dev_vbias);
    cudaFree(dev_dhbias);
    cudaFree(dev_dvbias);
    cudaFree(dev_tot_vones_temp);
    cudaFree(dev_tot_hones_temp);
    cudaFree(dev_tot_vones);
    cudaFree(dev_tot_hones);
  }
  
  /*This function propagates the visible units activation upwards to the hidden units*/
  DTYPE RBM::propup(DTYPE *v, DTYPE *w, DTYPE b) {
    DTYPE pre_sigmoid_activation = 0.0;
    // pre_sigmoid_activation = inner_product(w, w+n_visible, v, pre_sigmoid_activation);
    for(int j=0; j<n_visible; j++) {
      pre_sigmoid_activation += w[j] * v[j];
    }
    pre_sigmoid_activation += b;
    return (1.0 / (1.0 + exp(-pre_sigmoid_activation))); //sigmoid(pre_sigmoid_activation);
  }
  
  void RBM::reset_d_arrays() {
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
  
  
  /* Contrastive Divergence CUDA kernel*/
  void RBM::contrastive_divergence(int curr_i, DTYPE lr, DTYPE wc, DTYPE * dev_data) {
    reset_d_arrays();
    
    CUDA_CHECK(cudaMemcpy(dev_tot_vones_temp, &tot_vones_temp, sizeof(int),
			  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_tot_hones_temp, &tot_hones_temp, sizeof(int),
			  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_tot_vones     , &tot_vones     , sizeof(int),
			  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_tot_hones     , &tot_hones     , sizeof(int),
			  cudaMemcpyHostToDevice));
    //==============
    //Execute Kernel
    //==============
    
    if(batch_size > MAX_THREADS) {
      cerr << "ERROR: batch_size cannot exceed 1024" << endl;
    }
    
    // GET_TIME(k1_t1);
    // cerr << "time: " << k1_t1 << endl;
    int n_blocks = 1 + (batch_size - 1) / NUM_BATCH_THREADS_PER_BLOCK;
    cd_gpu <<< n_blocks, NUM_BATCH_THREADS_PER_BLOCK, 0, stream>>>
      (dev_data, curr_i, data_num_cols, dev_tot_vones_temp, dev_tot_hones_temp,
       dev_tot_vones, dev_tot_hones, dev_W, W_pitch,
       dev_hbias, dev_vbias, dev_curand_states,
       dev_ph_mean_batch, dev_nv_means_batch, dev_nh_means_batch,
       dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, curand_batch_width);
    cudaDeviceSynchronize();
    // GET_TIME(k1_t2);
    // cerr << "k1 time: " << get_duration(k1_t1, k1_t2) << endl;
    CUDA_CHECK(cudaGetLastError());
    
    // cerr << "Initiating write\n";
    dim3 num_blocks, num_threads;
    dims_to_num_threads_and_blocks(n_visible, n_hidden, num_blocks, num_threads);
    // GET_TIME(k2_t1);
    write_results_to_memory <<< num_blocks, num_threads, 0, stream>>>
      (dev_data, dev_W, W_pitch, lr, wc, dev_ph_mean_batch, dev_nv_means_batch,
       dev_nh_means_batch, dev_ph_sample_batch, dev_nv_samples_batch,
       dev_nh_samples_batch, dev_hbias, dev_vbias, dev_dhbias, dev_dvbias,
       data_num_rows, data_num_cols, curr_i);
    cudaDeviceSynchronize();
    // GET_TIME(k2_t2);
    // cerr << "k2 time: " << get_duration(k2_t1, k2_t2) << endl;
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(&tot_vones_temp, dev_tot_vones_temp, sizeof(int),
			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&tot_hones_temp, dev_tot_hones_temp, sizeof(int),
			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&tot_vones     , dev_tot_vones     , sizeof(int),
			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&tot_hones     , dev_tot_hones     , sizeof(int),
			  cudaMemcpyDeviceToHost));
  }
  
  void RBM::copy_matrices_to_host() {
    // cerr << "RBM baseline copy_matrices_to_host\n";

    CUDA_CHECK(cudaMemcpy2D(WArray, n_visible * sizeof(DTYPE), dev_W, W_pitch,
    			    n_visible * sizeof(DTYPE), n_hidden, cudaMemcpyDeviceToHost));
    arrayToMatrix(WArray, W, n_hidden, n_visible);
    CUDA_CHECK(cudaMemcpy(vbias, dev_vbias, n_visible * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbias, dev_hbias, n_hidden  * sizeof(DTYPE),
    			  cudaMemcpyDeviceToHost));
  }
  
  
  void RBM::reconstruct(DTYPE *v, DTYPE *reconstructed_v) {
    DTYPE *h = new DTYPE[n_hidden];
    DTYPE pre_sigmoid_activation;

    for(int i=0; i<n_hidden; i++) {
      h[i] = propup(v, W[i], hbias[i]);
    }

    for(int i=0; i<n_visible; i++) {
      pre_sigmoid_activation = 0.0;
      for(int j=0; j<n_hidden; j++) {
	pre_sigmoid_activation += W[j][i] * h[j];
      }
      pre_sigmoid_activation += vbias[i];
      reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
    }

    delete[] h;
  }
  
  void RBM::printParams() {
    cout << "\n";

    cout << "W is: \n";
    for(int i=0; i<5; i++) {
      for(int j=0; j<5; j++) {
        cout << W[i][j] << " ";
      }
      cout << "\n";
    }
    cout << "..." << endl;

    cout << "hbias is: \n";
    for(int i=0; i<5; i++) 
      cout << hbias[i] << "\n";
    cout << "..." << endl;
    cout << "vbias is: \n";
    for(int i=0; i<5; i++)
      cout << vbias[i] << "\n";
    cout << "..." << endl;
  }
  
  void RBM::saveWeightMatrix() {
    // cout << "baseline saveWeightMatrix" << endl;
    copy_matrices_to_host();
    matrixToArray (W, WArray, n_hidden, n_visible);
    string wFilename(MATRIX_FILENAME);
    saveMatrix(WArray, (size_t) n_hidden, (size_t) n_visible, wFilename);
    string hbiasFilename("dat_files/hbias.dat");
    saveArray(hbias, (size_t) n_hidden, hbiasFilename);
    string vbiasFilename("dat_files/vbias.dat");
    saveArray(vbias, (size_t) n_visible, vbiasFilename);
  }

  //NOTE: loadWeightMatrix is not complete
  // void RBM::loadWeightMatrix() {
  //   string filename(MATRIX_FILENAME);
  //   size_t num_rows;
  //   size_t num_cols;
  //   loadArray(WArray, num_rows, num_cols, filename);
  //   if((num_rows != n_hidden) || (num_cols != n_visible)) {
  //     cout << "WARNING: matrix found in " << filename << " has dimensions "
  // 	   << num_rows << " X " << num_cols << " while dimensions " <<
  // 	n_hidden << " X " << n_visible << " were expected from 
  //   }
  // }
  
  void RBM::printExpResult(DTYPE vones, DTYPE hones) {
    cout << "tot 1s in v: " <<tot_vones << "\n";
    cout << "tot 1s in h: " <<tot_hones << "\n";
    cout << "average 0s in v: " << (DTYPE)n_visible - vones << "\n";
    cout << "average 0s in h: " << (DTYPE)n_hidden - hones << "\n\n";
  }
}
