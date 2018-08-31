#ifndef INCLUDED_cpu_baseline
#define INCLUDED_cpu_baseline

#include <vector>
using namespace std;

namespace cpu_baseline{
  class RBM;
  void test_rbm(int, int, double, int, int, const char*, int, int, double);
}

class cpu_baseline :: RBM {

public:
  int N;
  int n_visible;
  int n_hidden;
  int batch_size;

  int *input;
  int last_k;
  
  int tot_vones_temp;
  int tot_hones_temp;
  int tot_vones;
  int tot_hones;
  double time1;
  double time2;
  double time3;
  double v_sample_time;
  double h_sample_time;

  double **W;
  double *hbias;
  double *vbias;
  double **dW;
  double *dhbias;
  double *dvbias;

  RBM(int, int, int, int, double**, double*, double*);
  ~RBM();
  void contrastive_divergence(int**, int, double, int, double);
  void sample_h_given_v(int*, double*, int*);
  void sample_v_given_h(int*, double*, int*);
  double propup(int*, double*, double);
  double propdown(int*, int, double);
  void gibbs_hvh(int*, double*, int*, double*, int*);
  void reconstruct(int*, double*);
  void printParams();
  void printExpResult(double, double, double);
};

#endif
