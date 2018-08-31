/* 
   This file is the implemetation of RBM following the original algorithm.
   It is used as the baseline algorithm for comparison.
*/

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <string.h>
#include "../include/cpu_baseline.h"
#include "../include/utils.h"
#include <fstream>

using namespace std;
using namespace utils;

#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28


namespace cpu_baseline {

  /* RBM constructor */
  RBM::RBM(int size, int n_v, int n_h, int b_size, double **w, double *hb, double *vb) {
    N = size;
    n_visible = n_v;
    n_hidden = n_h;
    batch_size = b_size;

    last_k = 0;
    /* parameters for debuging and developtment purpose */
    tot_vones = 0;
    tot_hones = 0;
    tot_vones_temp = 0;
    tot_hones_temp = 0;
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    v_sample_time = 0.0;
    h_sample_time = 0.0;
    /* end */

    /* initializing learning parameters */
    if(w == NULL) {
      W = new double*[n_hidden];
      for(int i=0; i<n_hidden; i++) W[i] = new double[n_visible];
      double a = 1.0 / n_visible;
      
      for(int i=0; i<n_hidden; i++) {
	for(int j=0; j<n_visible; j++) {
	  W[i][j] = uniform(-a, a);
	}
      }
    } else {
      W = w;
    }
    
    if(hb == NULL) {
      hbias = new double[n_hidden];
      for(int i=0; i<n_hidden; i++) hbias[i] = 0;
    } else {
      hbias = hb;
    }

    if(vb == NULL) {
      vbias = new double[n_visible];
      for(int i=0; i<n_visible; i++) vbias[i] = 0;
    } else {
      vbias = vb;
    }
  
    input = new int[n_visible];

    /* initialize gradient for updating the parameters when batch_size >= 1 */
    if(batch_size != 0) {
      dW = new double*[n_hidden];
      for(int i=0; i<n_hidden; i++) {
	dW[i] = new double[n_visible];
	for(int j=0; j<n_visible; j++) {
	  dW[i][j] = 0.0;
	}
      }

      dhbias = new double[n_hidden];
      for(int i=0; i<n_hidden; i++) dhbias[i] = 0;

      dvbias = new double[n_visible];
      for(int i=0; i<n_visible; i++) dvbias[i] = 0;
    }

  }

  /* RBM destructor */
  RBM::~RBM() {
    if(batch_size != 0) {
      for(int i=0; i<n_hidden; i++) {
	delete[] W[i];
	delete[] dW[i];
      }
      delete[] dW;
      delete[] dhbias;
      delete[] dvbias;
    }
    else {
      for(int i=0; i<n_hidden; i++) delete[] W[i];
    }

    delete[] W;
    delete[] hbias;
    delete[] vbias;
    delete[] input;
  }

  /* Contrastive Divergence*/
  void RBM::contrastive_divergence(int **data, int curr_i, double lr, int k, double wc) {
    // int *input = new int[n_visible];
    double *ph_mean = new double[n_hidden];
    int *ph_sample = new int[n_hidden];
    double *nv_means = new double[n_visible];
    int *nv_samples = new int[n_visible];
    double *nh_means = new double[n_hidden];
    int *nh_samples = new int[n_hidden];

    unsigned long long t11 = rdtsc();
    for (int cur_i=0; cur_i<batch_size; cur_i++) {
      input = data[curr_i*batch_size+cur_i];
      last_k = 0;

      /* CD-k */
      sample_h_given_v(input, ph_mean, ph_sample);

      /* step 0 to k-1 */
      for(int step=0; step<k-1; step++) {
      	if(step == 0) {
      	  gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
      	} else {
      	  gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
      	}
      }
      
      /* step k */
      last_k = 1;
      if(k == 1) {
      	gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
      } else {
      	gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
      }
      
      /* update the gradient based on current sample input */
      for(int i=0; i<n_hidden; i++) {
	for(int j=0; j<n_visible; j++) {
	  dW[i][j] += ph_mean[i] * input[j] - nh_means[i] * nv_samples[j];
	  // dW[i][j] += ph_mean[i] * input[j] - nh_samples[i] * nv_samples[j] ;
	}
	dhbias[i] += ph_mean[i] - nh_means[i];
	// dhbias[i] += ph_sample[i] - nh_samples[i];
      }
      for(int i=0; i<n_visible; i++) {
	dvbias[i] += input[i] - nv_samples[i];
      }
    }
    unsigned long long t12 = rdtsc();
    time1 += (double)(t12 - t11)/1000.0/3199987.0;
    // cout << "dvbias[0] = " << dvbias[0] << endl;
    unsigned long long t21 = rdtsc();
    for(int i=0; i<n_hidden; i++) {
      for(int j=0; j<n_visible; j++) {
	// W[i][j] += lr * dW[i][j] / batch_size; // no weight decay
	W[i][j] = W[i][j] + lr * (dW[i][j] / batch_size - wc * W[i][j]); // weight decay
	dW[i][j] = 0.0;
      }
      hbias[i] += lr * dhbias[i] / batch_size;
      dhbias[i] = 0.0;
    }
    
    for(int i=0; i<n_visible; i++) {
      vbias[i] += lr * dvbias[i] / batch_size;
      dvbias[i] = 0.0;
    }
    unsigned long long t22 = rdtsc();
    time2 += (double)(t22 - t21)/1000.0/3199987.0;

    delete[] ph_mean;
    delete[] ph_sample;
    delete[] nv_means;
    delete[] nv_samples;
    delete[] nh_means;
    delete[] nh_samples;
  }

  /*This function infers state of hidden units given visible units*/
  void RBM::sample_h_given_v(int *v0_sample, double *mean, int *sample) {
    double r;
 
    unsigned long long t1 = rdtsc();
    if(last_k == 0) {
      for(int i=0; i<n_hidden; i++) {
	r = rand() / (RAND_MAX + 1.0);
	mean[i] = propup(v0_sample, W[i], hbias[i]);
	sample[i] = r < mean[i];

	/* this part is for developing purpose*/
	if(sample[i] == 1) {
	  tot_hones_temp += 1;
	  tot_hones += 1;
	}
	/* end */
      }
    }
    else if(last_k == 1){
      for(int i=0; i<n_hidden; i++) {
	// r = rand() / (RAND_MAX + 1.0);
	mean[i] = propup(v0_sample, W[i], hbias[i]);
      }
    }
    unsigned long long t2 = rdtsc();
    h_sample_time += (double)(t2 - t1)/1000.0/3199987.0;
  }

  /*This function infers state of visible units given hidden units*/
  void RBM::sample_v_given_h(int *h0_sample, double *mean, int *sample) {
    double r;

    unsigned long long t1 = rdtsc();
    for(int i=0; i<n_visible; i++) {
      r = rand() / (RAND_MAX + 1.0);
      mean[i] = propdown(h0_sample, i, vbias[i]);
      sample[i] = r < mean[i];

      /* this part is for developing purpose*/
      if(sample[i] == 1) {
	tot_vones_temp += 1;
	tot_vones += 1;
      }
      /* end */
    }
    unsigned long long t2 = rdtsc();
    v_sample_time += (double)(t2 - t1)/1000.0/3199987.0;
  }

  /*This function propagates the visible units activation upwards to the hidden units*/
  double RBM::propup(int *v, double *w, double b) {
    double pre_sigmoid_activation = 0.0;
    // pre_sigmoid_activation = inner_product(w, w+n_visible, v, pre_sigmoid_activation);
    for(int j=0; j<n_visible; j++) {
      pre_sigmoid_activation += w[j] * v[j];
    }
    pre_sigmoid_activation += b;
    return 1.0 / (1.0 + exp(-pre_sigmoid_activation)); //sigmoid(pre_sigmoid_activation);
  }

  /*This function propagates the hidden units activation downwards to the visible units*/
  double RBM::propdown(int *h, int i, double b) {
    double pre_sigmoid_activation = 0.0;
    for(int j=0; j<n_hidden; j++) {
      pre_sigmoid_activation += W[j][i] * h[j];
    }
    pre_sigmoid_activation += b;
    return 1.0 / (1.0 + exp(-pre_sigmoid_activation)); //sigmoid(pre_sigmoid_activation);
  }

  /*This function implements one step of Gibbs sampling, starting from the hidden state*/
  void RBM::gibbs_hvh(int *h0_sample, double *nv_means, int *nv_samples, \
		      double *nh_means, int *nh_samples) {

    sample_v_given_h(h0_sample, nv_means, nv_samples);
    sample_h_given_v(nv_samples, nh_means, nh_samples);
  }

  void RBM::reconstruct(int *v, double *reconstructed_v) {
    double *h = new double[n_hidden];
    double pre_sigmoid_activation;

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
    for(int i=0; i<10; i++) {
      for(int j=0; j<10; j++) {
        cout << W[i][j] << " ";
      }
      cout << "\n";
    }
    cout << "..." << endl;

    cout << "hbias is: \n";
    for(int i=0; i<20; i++) 
      cout << hbias[i] << "\n";
    cout << "..." << endl;
    cout << "vbias is: \n";
    for(int i=0; i<20; i++)
      cout << vbias[i] << "\n";
    cout << "..." << endl;
  }

  void save_image(unsigned char * image, char * filename, int width, int height) {
    ofstream out(filename);
    out << "P6\n";
    out << width << ' ' << height << endl;
    out << "255" << endl;
    // unsigned char curr;
    // unsigned int count = 0;
    for(int i = 0; i < width * height; i++) {
      out << image[i] << image[i] << image[i];
      // if(count == 0) {
      //   curr = image[i];
      //   count = 1;
      // } else if(curr != image[i]) {
      //   out << count << endl;
      //   out << (int) curr << endl;
      //   count = 0;
      // } else {
      //   ++count;
      // }
    }
    // out << 0 << endl;
  }
  

  void RBM::printExpResult(double vones, double hones, double t) {
    cout << "tot 1s in v: " <<tot_vones << "\n";
    cout << "tot 1s in h: " <<tot_hones << "\n";
    cout << "average 0s in v: " << (double)n_visible - vones << "\n";
    cout << "average 0s in h: " << (double)n_hidden - hones << "\n\n";

    cout << "time1: " << time1 << " time2: " << time2 << " time3: 0" << "\n"; 
    cout << "h_sample_time: " << h_sample_time << "\n";
    cout << "v_sample_time: " << v_sample_time << "\n";
    // cout << "training time: " << t << "\n" << endl;
    
    // printParams();
  }


  void test_rbm(int train_N, int batch_size, double learning_rate, int k, int training_epochs, const char *dataset, int n_visible, int n_hidden, double weightcost) {
    srand(0);

    unsigned long long t1, t2, t11, t22;
    double t, t_epoch;
    double vones = 0.0;
    double hones = 0.0;
    int **train_X;

    int *datainfo;
    datainfo = new int[2]; // 0: datasize; 1: dataDimension
    findDimension(datainfo, dataset);
    cout << datainfo[0] << "  " << datainfo[1]<< endl;
    train_X = new int*[datainfo[0]];
    for(int i=0; i<datainfo[0]; i++) {
      train_X[i] = new int[datainfo[1]];
    }
    loadDataInt(train_X, dataset);
  
    // construct RBM
    RBM rbm(train_N, n_visible, n_hidden, batch_size, NULL, NULL, NULL);

    // train RBM
    cout << "begin training the RBM ..." << endl;
    t1 = rdtsc();
    
    GET_TIME(start_train_time);

    for(int epoch=0; epoch<training_epochs; epoch++) {
      cout << "epoch: " << epoch << endl;
      t11 = rdtsc();
      if(batch_size != 0) {
	int nbatch = train_N/batch_size;
	for(int i=0; i<nbatch; i++) {
	  // cout << "Starting batch: " << i << endl;
	  rbm.contrastive_divergence(train_X, i, learning_rate, k, weightcost);
	}
      }
      else {
	cout << "please indicate the batch size";
      }  
      t22 = rdtsc();
      t_epoch = (double)(t22-t11)/1000.0/3199987.0;
      vones += (double)rbm.tot_vones_temp/(train_N * training_epochs * k);
      hones += (double)rbm.tot_hones_temp/(train_N * training_epochs * (k+1));
      rbm.tot_vones_temp = 0;
      rbm.tot_hones_temp = 0;
      cout << "epoch " << epoch << ": " << t_epoch << "\n";
      rbm.printParams();
    }

    GET_TIME(end_train_time);
    

    t2 = rdtsc();
    t = (double)(t2-t1)/1000.0/3199987.0;
    cout << "\n";

    /* this is for development purposes */
    rbm.printExpResult(vones, hones, t);
    cout << "Training time: " << get_duration(start_train_time, end_train_time) << endl;
  }
}
