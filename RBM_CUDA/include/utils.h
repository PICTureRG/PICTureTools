#ifndef _UTILS_H
#define _UTILS_H 

#include <vector>
#include <string>
#include "constants.h"
/* #include <boost/date_time/posix_time/posix_time.hpp> */

/* #define GET_TIME(t) boost::posix_time::ptime t (boost::posix_time::microsec_clock::local_time()); */
#define GET_TIME(t) double t = get_wall_time();

#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MAX_UCHAR 255

//How many threads per block to run in the finish_sampling_kernel kernel. 
#define CUBLAS_CHECK(x) if (x != CUBLAS_STATUS_SUCCESS) cerr << "Cublas error: file = " << __FILE__ ", line number = " << __LINE__ << endl


using namespace std;
namespace utils{
  double get_wall_time();

  bool isBinary(int * array,int length);
  
  bool saveArray(const DTYPE* array, size_t length, const std::string& file_path);
  bool loadArray(DTYPE *& array, size_t & length, const std::string& file_path);

  bool saveMatrix(const DTYPE* matrix, size_t m, size_t n, const std::string& file_path);
  bool loadMatrix(DTYPE *& matrix, size_t & m, size_t & n, const std::string& file_path);
      
  void printMNIST(int * mnist_image);
  void save_image(DTYPE * image, char * filename, int width, int height);
  void save_image(int    * image, char * filename, int width, int height);

  /* DTYPE get_duration(boost::posix_time::ptime t1, boost::posix_time::ptime t2); */
  double get_duration(double t1, double t2);
  
  unsigned long long rdtsc();
  DTYPE uniform(DTYPE min, DTYPE max);
  int binomial(int n, DTYPE p);
  DTYPE randn(DTYPE mean, DTYPE sigma);
  DTYPE sigmoid(DTYPE x);
  int ReverseInt (int i);
  void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<vector<DTYPE> > &arr, const char *datafile);
  void ReadMNISTlabel(int NumberOfImages, vector<int> &arr, const char *datafile);
  void ReadCal101(int NumberOfImages, int DataOfAnImage, vector<vector<DTYPE> > &arr, const char *datafile);
  void ReadTxt(int NumberOfImages, int DataOfAnImage, vector<vector<DTYPE> > &arr, const char *datafile);
  void prepData(DTYPE**, DTYPE**, int, int);
  void loadDataInt(int**, const char*);
  void loadDataIntOrdered(int**, const char*);
  void loadDataReal(DTYPE**, const char*);
  void findDimension(int*, const char*);
}
#endif
