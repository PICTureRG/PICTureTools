#ifndef INCLUDED_rbm_baseline
#define INCLUDED_rbm_baseline

#include <vector>
#include <curand_kernel.h>
#include <iostream>
/* #include <boost/date_time/posix_time/posix_time.hpp> */
#include "utils.h"
#include "cuda_utils.h"

#include "constants.h"


using namespace std;
using namespace utils;

//if defined, executes two kernels in two streams:
/* #define SIMULTANEOUS_EXECUTION */

namespace baseline{
  class RBM;
  /* void test_rbm(int, int, DTYPE, int, int, const char*, int, int, DTYPE); */
}

class baseline::RBM {

public:
  curandState_t * dev_curand_states;
  cudaStream_t stream;
  
  int N;
  int n_visible;
  int n_hidden;
  int batch_size;
  
  /* int *input; */
  int * data;
  int data_num_rows;
  int data_num_cols;
  int last_k;
  int k;
  
  int tot_vones_temp;
  int tot_hones_temp;
  int tot_vones;
  int tot_hones;
  // DTYPE time1;
  // DTYPE time2;
  // DTYPE time3;
  // DTYPE v_sample_time;
  // DTYPE h_sample_time;
  
  DTYPE **W;
  DTYPE * WArray;//Need 1D array for CUDA.
  DTYPE *hbias;
  DTYPE *vbias;
  /* DTYPE **dW; */
  /* DTYPE * dWArray; */
  DTYPE *dhbias;
  DTYPE *dvbias;

  DTYPE *dev_ph_mean_batch;
  DTYPE *dev_nv_means_batch;
  DTYPE *dev_nh_means_batch;
  DTYPE *dev_ph_sample_batch;
  DTYPE *dev_nv_samples_batch;
  DTYPE *dev_nh_samples_batch;

  DTYPE * dev_W;
  size_t W_pitch;
  
  /* DTYPE * dev_dW; */
  /* size_t dW_pitch; */

  /* int * dev_data; */
  DTYPE * dev_hbias;
  DTYPE * dev_vbias;
  DTYPE * dev_dhbias;
  DTYPE * dev_dvbias;

  int * dev_tot_vones_temp;
  int * dev_tot_hones_temp;
  int * dev_tot_vones;
  int * dev_tot_hones;
  
  int curand_batch_width;

  void allocate_special_memory();
  void printParams();
  
  void reset_d_arrays();

  RBM(int, int, int, int, int, DTYPE**, DTYPE*, DTYPE*, int, int);
  ~RBM();
  void contrastive_divergence(int, DTYPE, DTYPE, DTYPE*);
  virtual void copy_matrices_to_host();
  
  /* void sample_h_given_v(int*, DTYPE*, int*); */
  /* void sample_v_given_h(int*, DTYPE*, int*); */
  DTYPE propup(DTYPE*, DTYPE*, DTYPE);
  /* DTYPE propdown(int*, int, DTYPE); */
  /* void gibbs_hvh(int*, DTYPE*, int*, DTYPE*, int*); */
  
  void reconstruct(DTYPE*, DTYPE*);
  void printExpResult(DTYPE, DTYPE);
  void saveWeightMatrix();

  /* static void test_rbm(int train_N, int batch_size, DTYPE learning_rate, int k, */
  /* 		       int training_epochs, const char *dataset, int n_visible, */
  /* 		       int n_hidden, DTYPE weightcost); */
  template<class RBM_Class>
    static void test_rbm(int train_N, int batch_size, DTYPE learning_rate, int k,
			 int training_epochs, const char *dataset, int n_visible,
			 int n_hidden, DTYPE weightcost, int num_streams) {
    GET_TIME(start_time);
    srand(0);
    /* DTYPE vones = 0.0; */
    /* DTYPE hones = 0.0; */
    int **train_X;

    int datainfo[2];
    // datainfo = new int[2]; // 0: datasize; 1: dataDimension
    findDimension(datainfo, dataset);
    /* cout << datainfo[0] << "  " << datainfo[1]<< endl; */
    train_X = new int*[datainfo[0]];
    for(int i=0; i<datainfo[0]; i++) {
      train_X[i] = new int[datainfo[1]];
    }
    cout << "Loading MNIST data\n";
    loadDataInt(train_X, dataset);
    int data_len = datainfo[0] * datainfo[1];
    int * train_X_Array = new int[data_len];
    matrixToArrayInt(train_X, train_X_Array, datainfo[0], datainfo[1]);
    if(!isBinary(train_X_Array, data_len)) {
      cerr << "ERROR: Data not in binary format\n";
      return;
    }
    
    DTYPE * train_X_Array_doubles = new DTYPE[data_len];
    for(int i = 0; i < data_len; i++) {
      train_X_Array_doubles[i] = (DTYPE) train_X_Array[i];
    }
    
    //Allocate shared input/data array
    DTYPE * dev_data;
    CUDA_CHECK(cudaMalloc((void**)&dev_data, data_len * sizeof(DTYPE)));
    CUDA_CHECK(cudaMemcpy(dev_data, train_X_Array_doubles, data_len * sizeof(DTYPE),
			  cudaMemcpyHostToDevice));

    RBM_Class ** rbms = new RBM_Class*[num_streams];
    /* cudaStream_t * streams = new cudaStream_t[num_streams]; */
    for(int i = 0; i < num_streams; i++) {
      cout << "Constructing RBM " << (i+1) << "/" << num_streams << endl;
      rbms[i] = new RBM_Class(train_N, n_visible, n_hidden, batch_size, k, NULL, NULL, NULL,
			      datainfo[0], datainfo[1]);
      rbms[i]->allocate_special_memory();
      /* cudaStreamCreate(&streams[i]); */
    }

    //Allocate space for cublas handle
    /* #ifdef SIMULTANEOUS_EXECUTION */
    /* cout << "Constructing RBM 2 for SIMULTANEOUS EXECUTION!\n"; */
    /* RBM_Class rbm2(train_N, n_visible, n_hidden, batch_size, k, NULL, NULL, NULL, */
    /* 		   datainfo[0], datainfo[1]); */
    /* rbm2.allocate_special_memory(); */
    /* #endif */
    
    // printMNIST(train_X_Array);
    // printMNIST(&train_X_Array[784]);
    /* cout << "data dimension: " << datainfo[1] << endl; */
    
    // train RBM
    cout << "number of training epochs: " << training_epochs << endl;
    cout << "Training RBM ..." << endl;
    GET_TIME(start_train_time);
    for(int epoch=0; epoch<training_epochs; epoch++) {
      /* cout << "epoch: "; */
      if(batch_size != 0) {
	int nbatch = train_N/batch_size;
	for(int i=0; i<nbatch; i++) {
	  for(int stream_idx = 0; stream_idx < num_streams; stream_idx++) {
	    rbms[stream_idx]->contrastive_divergence(i, learning_rate, weightcost,
						     dev_data);
	  }
          /* #ifdef SIMULTANEOUS_EXECUTION */
	  /* rbm2.contrastive_divergence(i, learning_rate, weightcost, dev_data); */
	  /* #endif */
	}
      } else {
	cout << "please indicate the batch size";
      }
      /* cout << "Finished epoch " << epoch << endl; */
      for(int stream_idx = 0; stream_idx < num_streams; stream_idx++) {
      	/* rbms[stream_idx]->copy_matrices_to_host(); */
      	/* rbms[stream_idx]->printParams(); */
      }
      /* #ifdef SIMULTANEOUS_EXECUTION */
      /* cout << "RBM1 Paramaters" << endl */
      /* 	   << "===============" << endl; */
      /* #endif */
      /* rbm.copy_matrices_to_host(); */
      /* rbm.printParams(); */
      /* #ifdef SIMULTANEOUS_EXECUTION */
      /* cout << "RBM2 Paramaters" << endl */
      /* 	   << "===============" << endl; */
      /* rbm2.copy_matrices_to_host(); */
      /* rbm2.printParams(); */
      /* #endif */
    }
    for(int stream_idx = 0; stream_idx < num_streams; stream_idx++) {
      rbms[stream_idx]->copy_matrices_to_host();
      rbms[stream_idx]->printParams();
    }
    
    GET_TIME(end_train_time);
    cout << "Training time: " << get_duration(start_train_time, end_train_time) << endl;
    /* this for for develop purpose */
    // rbm.printExpResult(vones, hones, t);
    /* rbm.printExpResult(vones, hones); */
    /* rbms[0]->saveWeightMatrix(); */
    
    /* double * reconstructed = new double[n_visible]; */
    
    /* rbm.reconstruct(train_X[2], reconstructed); */
    /* save_image(train_X[2], (char*) "mnist_image_2.ppm", */
    /* 	       MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT); */
    /* save_image(reconstructed, (char*) "mnist_recon_2.ppm", */
    /* 	       MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT); */
    /* delete[] reconstructed; */

    for(int i = 0; i < datainfo[0]; i++) {
      delete[] train_X[i];
    }
    for(int i = 0; i < num_streams; i++) {
      delete rbms[i];
      /*   cudaStreamDestroy(streams[i]); */
    }
    delete[] rbms;
    /* delete[] streams; */
    
    delete[] train_X;
    delete[] train_X_Array;
    delete[] train_X_Array_doubles;
    /* #ifdef SIMULTANEOUS_EXECUTION */
    /* cudaStreamDestroy(rbm_stream); */
    /* cudaStreamDestroy(rbm2_stream); */
    /* #endif */
    cudaFree(dev_data);

    GET_TIME(end_time);
    cout << "Wall-clock execution time: " << get_duration(start_time, end_time) << " seconds" << endl;
  }

};


#endif
