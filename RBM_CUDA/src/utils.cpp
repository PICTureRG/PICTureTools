#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <vector>
#include <sys/time.h>
#include "../include/utils.h"
#include "../include/constants.h"

#include <fstream>

using namespace std;

#define NUM_BITS_IN_BYTE 8

namespace utils {

  bool isBinary(int * array,int length) {
    for(int i = 0; i < length; i++) {
      if((array[i] != 0) && (array[i] != 1)) return false;
    }
    return true;
  }


  double get_wall_time() {
    struct timeval time;
    gettimeofday(&time,NULL);
    return ((double) time.tv_sec) + ((double) time.tv_usec * 0.000001);
  }
  
  double get_duration(double t1, double t2) {
    return (t2 - t1);
  }
  

  bool saveArray(const DTYPE* array, size_t length, const std::string& file_path)
  {
    std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
    if ( !os.is_open() )
      return false;
    os.write(reinterpret_cast<const char*>(&length), std::streamsize(sizeof(size_t)));
    os.write(reinterpret_cast<const char*>(array), std::streamsize(length * sizeof(DTYPE)));
    os.close();
    return true;
  }

  bool loadArray(DTYPE *& array, size_t & length, const std::string& file_path)
  {
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if ( !is.is_open() )
      return false;
    is.read(reinterpret_cast<char*>(&length), std::streamsize(sizeof(size_t)));
    array = new DTYPE[length];
    is.read(reinterpret_cast<char*>(array), std::streamsize(length * sizeof(DTYPE)));
    is.close();
    return true;
  }


  bool saveMatrix(const DTYPE* matrix, size_t m, size_t n, const std::string& file_path)
  {
    std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
    if ( !os.is_open() )
      return false;
    os.write(reinterpret_cast<const char*>(&m), std::streamsize(sizeof(size_t)));
    os.write(reinterpret_cast<const char*>(&n), std::streamsize(sizeof(size_t)));
    os.write(reinterpret_cast<const char*>(matrix), std::streamsize(m * n * sizeof(DTYPE)));
    os.close();
    return true;
  }

  bool loadMatrix(DTYPE *& matrix, size_t & m, size_t & n, const std::string& file_path)
  {
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if ( !is.is_open() )
      return false;
    is.read(reinterpret_cast<char*>(&m), std::streamsize(sizeof(size_t)));
    is.read(reinterpret_cast<char*>(&n), std::streamsize(sizeof(size_t)));
    matrix = new DTYPE[m * n];
    is.read(reinterpret_cast<char*>(matrix), std::streamsize(m * n * sizeof(DTYPE)));
    is.close();
    return true;
  }

  //PRE: mnist_image should have length 784 and contain only 1s and 0s.
  //POST: Prints array in ASCII art format (28 X 28 image). 
  void printMNIST(int * mnist_image) {
    for(int i = 0; i < 784; i++) {
      if(i % 28 == 0) cout << endl;
      if(mnist_image[i] == 0) {
	cout << ' ';
      } else if(mnist_image[i] == 1) {
	cout << 1;
      } else {
	cout << "ERROR Processing MNIST image!" << endl;
      }
      cout << ' ';
    }
    cout << endl;
  }

  //PRE: image must contain width * height DTYPE values between 0 and 1. 
  void save_image(DTYPE * image, char * filename, int width, int height) {
    ofstream out(filename);
    out << "P6\n";
    out << width << ' ' << height << endl;
    out << "255" << endl;
    for(int i = 0; i < width * height; i++) {
      unsigned char pixel = (unsigned char) (image[i] * MAX_UCHAR);
      out << pixel << pixel << pixel;
    }
    out.close();
  }

  //PRE: image must contain width * height ints which are 0 or 1. 
  void save_image(int * image, char * filename, int width, int height) {
    ofstream out(filename);
    out << "P6\n";
    out << width << ' ' << height << endl;
    out << "255" << endl;
    for(int i = 0; i < width * height; i++) {
      unsigned char pixel = image[i] == 0 ? 0 : MAX_UCHAR;
      out << pixel << pixel << pixel;
    }
    out.close();
  }


  // double get_duration(boost::posix_time::ptime t1, boost::posix_time::ptime t2) {
  //   boost::posix_time::time_duration duration = t2 - t1;
  //   long msec = duration.total_microseconds();
  //   double sec = msec / 1000000.0;
  //   return(sec);
  // }
  
  unsigned long long rdtsc() {
    unsigned a, d;
    __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
    return ((unsigned long long) a) | (((unsigned long long) d) << 32);
  }

  DTYPE uniform(DTYPE min, DTYPE max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
  }

  int binomial(int n, DTYPE p) {
    if(p < 0 || p > 1) return 0;
      
    int c = 0;
    DTYPE r;
      
    for(int i=0; i<n; i++) {
      r = rand() / (RAND_MAX + 1.0);
      if (r < p) c++;
    }
    return c;
  }

  DTYPE randn(DTYPE mu, DTYPE sigma) {
    // random_device rd;
    // mt19937 gen(rd());
    // normal_distribution<> d(mu, sigma);
    // DTYPE randnum = d(gen);

    // return randnum;
    DTYPE U1, U2, W, mult, randnum;
    static DTYPE X1, X2;
    static int call = 0;

    if (call == 1) {
      call = !call;
      randnum = (mu + sigma * (DTYPE)X2);
      // cout << randnum << endl;
      return randnum;
    }
    do {
      U1 = -1 + ((DTYPE) rand() / RAND_MAX) * 2;
      U2 = -1 + ((DTYPE) rand() / RAND_MAX) * 2;
      W = U1 * U1 + U2 * U2;
    }
    while (W >= 1 || W == 0);

    mult = sqrt((-2 * log(W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;
    randnum = (mu + sigma * (DTYPE) X1);

    // cout << randnum << endl;
    return randnum;
  }

  DTYPE sigmoid(DTYPE x) {
#ifdef USING_DOUBLES
    return 1.0 / (1.0 + exp(-x));
#else
    return 1.f / (1.f + exp(-x));
#endif
  }
  
  int ReverseInt (int i)
  {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
  }

  /*https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set*/
  void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<vector<DTYPE> > &arr, const char *datafile)
  {
    arr.resize(NumberOfImages,vector<DTYPE>(DataOfAnImage));
    ifstream file(datafile,ios::binary);
    if (file.is_open())
      {
	int magic_number=0;
	int number_of_images=0;
	int n_rows=0;
	int n_cols=0;
	file.read((char*)&magic_number,sizeof(magic_number));
	magic_number= ReverseInt(magic_number);
	file.read((char*)&number_of_images,sizeof(number_of_images));
	number_of_images= ReverseInt(number_of_images);
	file.read((char*)&n_rows,sizeof(n_rows));
	n_rows= ReverseInt(n_rows);
	file.read((char*)&n_cols,sizeof(n_cols));
	n_cols= ReverseInt(n_cols);
	for(int i=0;i<number_of_images;++i)
	  {
	    for(int r=0;r<n_rows;++r)
	      {
		for(int c=0;c<n_cols;++c)
		  {
		    unsigned char temp=0;
		    file.read((char*)&temp,sizeof(temp));
		    arr[i][(n_rows*r)+c]= (DTYPE)temp;
		  }
	      }
	  }
      }
    else {
      cout << "\n!!!!!!!!!!!!!!!!! ERROR: file open failed\n" << endl;
      exit(-1);
    }
  }

  /*https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set*/
  void ReadMNISTlabel(int NumberOfImages, vector<int> &arr, const char *datafile)
  {
    arr.resize(NumberOfImages);
    ifstream file(datafile,ios::binary);
    if (file.is_open())
      {
	int magic_number=0;
	int number_of_images=0;
	file.read((char*)&magic_number,sizeof(magic_number));
	magic_number= ReverseInt(magic_number);
	file.read((char*)&number_of_images,sizeof(number_of_images));
	number_of_images= ReverseInt(number_of_images);
	for(int i=0;i<number_of_images;++i)
	  {
	    unsigned char temp=0;
	    file.read((char*)&temp,sizeof(temp));
	    arr[i]= (int)temp;
	  }
      }
    else {
      cout << "\n!!!!!!!!!!!!!!!!! ERROR: file open failed\n" << endl;
      exit(-1);
    }
  }

  /* CAL101:  https://people.cs.umass.edu/~marlin/data.shtml */
  /* Olivetti:  http://www.cs.nyu.edu/~roweis/data.html */
  /* Abalone:  http://archive.ics.uci.edu/ml/datasets/Abalone */
  /* 20Newsgroup:  http://www.cs.nyu.edu/~roweis/data.html */
  void ReadTxt(int NumberOfImages, int DataOfAnImage, vector<vector<DTYPE> > &arr, const char *datafile)
  {
    arr.resize(NumberOfImages,vector<DTYPE>(DataOfAnImage));
    ifstream file(datafile);
    if(file.is_open())
      {
	for(int i=0; i<NumberOfImages; ++i)
	  {
	    for(int j=0; j<DataOfAnImage; ++j)
	      {
		file >> arr[i][j];
	      }
	  }
      }
    else
      {
	cout << "\n!!!!!!!!!!!!!!!!! ERROR: file open failed\n" << endl;
	exit(-1);
      }
    file.close();
  }

  void ReadCal101(int NumberOfImages, int DataOfAnImage, vector<vector<DTYPE> > &arr, const char *datafile)
  {
    arr.resize(NumberOfImages,vector<DTYPE>(DataOfAnImage));
    ifstream file(datafile);
    if(file.is_open())
      {
	for(int i=0; i<NumberOfImages; ++i)
	  {
	    for(int j=0; j<DataOfAnImage; ++j)
	      {
		file >> arr[i][j];
	      }
	  }
      }
    else
      {
	cout << "File could not be opened." << endl;
      }
    file.close();
  }


  void prepData(DTYPE **data, DTYPE **output, int NumberOfImages, int DataOfAnImage) {
    DTYPE *mean;
    DTYPE *std;
    DTYPE *sum;

    sum = new DTYPE[DataOfAnImage];
    mean = new DTYPE[DataOfAnImage];
    std = new DTYPE[DataOfAnImage];
    for(int i=0; i<DataOfAnImage; i++) {
      sum[i] = 0.0;
      for(int j=0; j<NumberOfImages; j++)  {
	sum[i] += data[j][i];
      }
      mean[i] = sum[i]/NumberOfImages;
      sum[i] = 0.0;
      for(int j=0; j<NumberOfImages; j++)  {
	sum[i] += (data[j][i]-mean[i]) * (data[j][i]-mean[i]); 
      }
      std[i] = sqrt(sum[i]/(NumberOfImages-1));   
    }

    for(int i=0; i<NumberOfImages; i++) {
      for(int j=0; j<DataOfAnImage; j++) {
	if(std[j] != 0) {
	  output[i][j] = (data[i][j] - mean[j])/std[j];
	}
	else {
	  output[i][j] = data[i][j] - mean[j];
	}
      }
    }

    delete[] mean;
    delete[] std;
    delete[] sum;
  }


  void findDimension(int * datainfo, const char *dataset) {
    if(strcmp(dataset,"mnist") == 0) {
      datainfo[0] = 60000;
      datainfo[1] = 784;
    }
    else if(strcmp(dataset, "flipmnist") == 0) {
      datainfo[0] = 60000;
      datainfo[1] = 784;
    }
    else if(strcmp(dataset, "cal101") == 0) {
      datainfo[0] = 4100;
      datainfo[1] = 784;
    }
    else if(strcmp(dataset, "olivetti") == 0) {
      datainfo[0] = 400;
      datainfo[1] = 4096;
    }
    else if(strcmp(dataset, "micro-norb") == 0) {
      datainfo[0] = 19440;
      datainfo[1] = 1024;
    }
    else if(strcmp(dataset, "newsgroup") == 0) {
      datainfo[0] = 16242;
      datainfo[1] = 100;
    }
    else if(strcmp(dataset, "cbcl") == 0) {
      datainfo[0] = 6977;
      datainfo[1] = 361;
    }
    else if(strcmp(dataset, "abalone") == 0) {
      datainfo[0] = 4177;
      datainfo[1] = 8;
    }
    else if(strcmp(dataset, "financial") == 0) {
      datainfo[0] = 5135;
      datainfo[1] = 13;
    }
    // cout <<"input data dimension is: " << datainfo[0]<< " " << datainfo[1] << endl;
  }


  void loadDataInt(int **train_X, const char *dataset) {
    DTYPE **data;

    if(strcmp(dataset,"mnist") == 0) {
      char datafile[] = "data/mnist/train-images-idx3-ubyte";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadMNIST(60000,784,ar,datafile);

      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      DTYPE temp;
      DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  temp = data[i][j]/255.;
	  r_temp = rand() / (RAND_MAX + 1.0);
	  train_X[i][j] = r_temp < temp;
	}
      }

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "flipmnist") == 0) {
      char datafile[] = "data/mnist/train-images-idx3-ubyte";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadMNIST(60000,784,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      DTYPE temp;
      DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  temp = data[i][j]/255.;
	  r_temp = rand() / (RAND_MAX + 1.0);
	  train_X[i][j] = r_temp < temp;
	  train_X[i][j] = (int)(1-train_X[i][j]);
	}
      }

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "cal101") == 0) {
      char datafile[] = "data/cal101/cal28_train.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(4100,784,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  train_X[i][j] = (int) data[i][j];
	}
      }

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "newsgroup") == 0) {
      char datafile[] = "data/20Newsgroup/news.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(16242,100,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  train_X[i][j] = (int) data[i][j];
	}
      }
      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "micro-norb") == 0) {
      char datafile[] = "data/mnorb/micro-norb.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(19440,1024,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  train_X[i][j] = (int) data[i][j];
	}
      }
      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "abalone") == 0) {
      char datafile[] = "data/abalone/abalone.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(4177,8,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      DTYPE temp;
      DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  temp = data[i][j];
	  if(temp > 0.5) {
	    train_X[i][j] = 1;
	  }
	  else if(temp < 0.5) {
	    train_X[i][j] = 0;
	  }
	  else if(temp == 0.5) {
	    r_temp = rand() / (RAND_MAX + 1.0);
	    train_X[i][j] = r_temp < temp;
	  }
	}
      }
      // for (int i =0; i<2; i++) {
      //     for(int j = 0; j<8; j++) {
      //         cout << train_X[i][j] << endl;
      //     }
      // }
      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
  }

  void loadDataReal(DTYPE **train_X, const char *dataset) {
    DTYPE **data;

    if(strcmp(dataset,"mnist") == 0) {
      char datafile[] = "data/mnist/train-images-idx3-ubyte";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadMNIST(60000,784,ar,datafile);

      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      prepData(data, train_X, 60000, 784);

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "flipmnist") == 0) {
      char datafile[] = "data/mnist/train-images-idx3-ubyte";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadMNIST(60000,784,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      for (unsigned i=0; i<ar.size(); i++) {
	for (unsigned j=0; j<ar.at(i).size(); j++) { 
	  data[i][j] = 255.0 - data[i][j];
	}
      }
      prepData(data, train_X, 60000, 784); 

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "cal101") == 0) {
      char datafile[] = "data/cal101/cal28_train.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(4100,784,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      prepData(data, train_X, 4100, 784);

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "abalone") == 0) {
      char datafile[] = "data/abalone/abalone.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(4177,8,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      prepData(data, train_X, 4177, 8);

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "cbcl") == 0) {
      char datafile[] = "data/cbcl/cbcl.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(6977,361,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }
      // DTYPE temp;
      // DTYPE r_temp;
      prepData(data, train_X, 6977, 361);

      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "financial") == 0) {
      char datafile[] = "data/financial/financial.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(5135,13,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }
      // DTYPE temp;
      // DTYPE r_temp;
      prepData(data, train_X, 5135, 13);
        
      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
    else if(strcmp(dataset, "olivetti") == 0) {
      char datafile[] = "data/olivetti/olivetti.txt";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadTxt(400,4096,ar,datafile);
      data = new DTYPE*[ar.size()];
      for (unsigned i=0; i<ar.size(); i++) {
	data[i] = new DTYPE[ar.at(i).size()];
	copy(ar.at(i).begin(), ar.at(i).end(), data[i]);
      }

      // DTYPE temp;
      // DTYPE r_temp;
      prepData(data, train_X, 400, 4096);
        
      for (int i=0; i<ar.size(); i++) {
	delete[] data[i];
      }    
      delete[] data;
    }
  }

  void loadDataIntOrdered(int **train_X, const char *dataset) {
    if(strcmp(dataset,"mnist") == 0) {
      char datafile[] = "data/mnist/train-images-idx3-ubyte";
      char labelfile[] = "data/mnist/train-labels-idx1-ubyte";
      /*load training data*/
      vector<vector<DTYPE> > ar;
      ReadMNIST(60000,784,ar,datafile);
      /* load training data label */
      vector<int> arr;
      ReadMNISTlabel(60000,arr,labelfile);

      /* store the indexes of each 0-9 digit numbers*/
      vector<vector<int> > digit;
      digit.resize(10, vector<int>(0));
      for (int i=0; i<60000; i++) {
	switch (arr[i]) {
	case 0: digit[0].push_back(i); break;
	case 1: digit[1].push_back(i); break;
	case 2: digit[2].push_back(i); break;
	case 3: digit[3].push_back(i); break;
	case 4: digit[4].push_back(i); break;
	case 5: digit[5].push_back(i); break;
	case 6: digit[6].push_back(i); break;
	case 7: digit[7].push_back(i); break;
	case 8: digit[8].push_back(i); break;
	case 9: digit[9].push_back(i); break;
	}
      }

      /* print out the amount of each digit number */
      // int size = 0;
      // for (int i=0; i<10; i++) {
      //   cout << digit[i].size() << endl;
      //   size += digit[i].size();
      // }
      // cout << size << endl;

      /* load ordered data into train_X */
      DTYPE temp;
      DTYPE r_temp;
      int count = 0;
      for (int i=0; i<10; i++) {
	for (int j=0; j<digit[i].size(); j++) {
	  for (int k=0; k<ar.at(count).size(); k++) {
	    temp = ar[digit[i][j]][k]/255.;
	    r_temp = rand() / (RAND_MAX + 1.0);
	    train_X[count][k] = r_temp < temp;
	  }
	  count++;
	}
      }   
       
    }
  }

}
