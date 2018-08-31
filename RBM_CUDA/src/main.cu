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

#include "../include/rbm_baseline.h"
#include "../include/rbm_delta_product.h"
#include "../include/rbm_dynamic_cublas.h"
#include "../include/rbm_matrix.h"
#include "../include/rbm_dp_matrix.h"
#include "../include/cpu_baseline.h"

// #include "../include/rbm_blas.h" //////////////////////////////blas not installed
// #include "../include/rbm_opt.h"
// #include "../include/rbm_diff.h"
// #include "../include/rbm_diff_more.h"
// #include "../include/rbm_diff_blas.h"
// #include "../include/rbm_baseline_bi.h"
// #include "../include/rbm_diff_bi.h"
// #include "../include/rbm_diff_more_bi.h"
// #include "../include/rbm_diff_ordered.h"
// #include "../include/rbm_opt_diff_v1.h"
// #include "../include/rbm_opt_diff_v2.h"
// #include "../include/rbm_opt_diff_v3.h"
// #include "../include/rbm_opt_diff_v4.h"
// #include "../include/rbm_triangular_v1.h"
// #include "../include/rbm_triangular_clustered.h"
// #include "../include/rbm_triangular_group.h"
// #include "../include/rbm_triangular_v1_norm.h"
// #include "../include/rbm_triangular_clustered_norm.h"
// #include "../include/rbm_triangular_group_norm.h"
// #include "../include/rbm_comb.h"
// #include "../include/rbm_ti_ws.h"
// #include "../include/rbm_pseudo.h"
#include "../include/utils.h"
using namespace std;
using namespace utils;


int main(int argc, char* argv[]) {
  if(argc != 11 && argc != 12) {
    cout << "ERROR: ./rbm <train_N> <batchsize> <learning_rate> <k> <epoch> <filename> <option> <n_visible> <n_hidden> <weightcost>" << endl;
    return -1;
  }

  int train_N = atoi(argv[1]);
  int batchsize = atoi(argv[2]);
  DTYPE learning_rate = atof(argv[3]);
  int myk = atoi(argv[4]);
  int epoch = atoi(argv[5]);
  char *filename = argv[6];
  char *option = argv[7];
  int n_visible = atoi(argv[8]);
  int n_hidden = atoi(argv[9]);
  DTYPE weightcost = atof(argv[10]);
  int num_streams = 1;
  if(argc > 11) num_streams = atoi(argv[11]);
  
  cout << "This is the result of the " << option << " implemetations of RBM.\n\n";

  cout << "train_N is: " << train_N << "\n";
  cout << "batch size is: " << batchsize << "\n";
  cout << "the learning rate is: " << learning_rate << "\n";
  cout << "gibbs steps k is: " << myk << "\n";
  cout << "epoch is: " << epoch << "\n";
  cout << "dataset is: " << filename << "\n";
  cout << "Optimization option is: " << option << "\n";
  cout << "n_visible is: " << n_visible << "\n";
  cout << "n_hidden is: " << n_hidden << "\n";
  cout << "weightcost is: " << weightcost << "\n";
  cout << "num_streams is: " << num_streams << "\n\n";
  
  if(strcmp(option,"baseline") == 0) {
    baseline::RBM::test_rbm<baseline::RBM>(train_N, batchsize, learning_rate, myk, epoch, filename, n_visible, n_hidden, weightcost, num_streams);
  } else if(strcmp(option,"cpu") == 0) {
    cpu_baseline::test_rbm(train_N, batchsize, learning_rate, myk, epoch, filename, n_visible, n_hidden, weightcost);
  } else if(strcmp(option, "matrix") == 0) {
    matrix::RBM_matrix::test_rbm<matrix::RBM_matrix>(train_N, batchsize, learning_rate, myk, epoch, filename, n_visible, n_hidden, weightcost, num_streams);
  } else if(strcmp(option, "delta_product") == 0) {
    delta_product::RBM_delta::test_rbm<delta_product::RBM_delta>(train_N, batchsize, learning_rate, myk, epoch, filename, n_visible, n_hidden, weightcost, num_streams);
  } else if(strcmp(option, "dp_matrix") == 0) {
    dp_matrix::RBM_dp_matrix::test_rbm<dp_matrix::RBM_dp_matrix>(train_N, batchsize, learning_rate, myk, epoch, filename, n_visible, n_hidden, weightcost, num_streams);
  } else if(strcmp(option, "dynamic_cublas") == 0) {
    dynamic_cublas::RBM_dynamic_cublas::test_rbm<dynamic_cublas::RBM_dynamic_cublas>(train_N, batchsize, learning_rate, myk, epoch, filename, n_visible, n_hidden, weightcost, num_streams);
  } else {
    cout << "ERROR: Arg 7 should be one of {cpu, matrix, dp_matrix}" << endl;
  }
  return 0;
}

