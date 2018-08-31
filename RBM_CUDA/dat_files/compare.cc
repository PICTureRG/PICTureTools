#include <iostream>
#include "../include/utils.h"
#include "../include/constants.h"

//This code is designed to be compiled independently.
//It compares .dat files to find differences between them. 

using namespace std;
using namespace utils;

DTYPE abs(DTYPE d) {
  if(d < 0.0) return -d;
  return d;
}

int main(int argc, char ** argv) {
  if(argc < 3) {
    cout << "Need 2 dat files." << endl;
  } else {
    DTYPE * array_1;
    DTYPE * array_2;
    size_t length;
    // string fn1 = "../array1.dat";
    // string fn2 = "../array2.dat";
    string fn1 = argv[1];
    string fn2 = argv[2];
    if(!loadArray(array_1, length, fn1)) {
      cerr << "error: filename " << fn1 << " was not found\n";
    }
    if(!loadArray(array_2, length, fn2)) {
      cerr << "error: filename " << fn2 << " was not found\n";
    }
    int same = 0;
    int diff = 0;
    DTYPE max_diff;
    bool max_diff_defined = false;
    
    for(int i = 0; i < length; i++) {
      if(array_1[i] == array_2[i]) same++;
      else {
	diff++;
	DTYPE curr_diff = abs(array_2[i] - array_1[i]);
	cout << curr_diff << endl;
	if(max_diff_defined) {
	  if(curr_diff > max_diff) {
	    max_diff = curr_diff;
	  }
	} else {
	  max_diff_defined = true;
	  max_diff = curr_diff;
	}
      }
      
      // if(abs(array_1[i] - array_2[i]) > 0.0001) {
      // 	cout << "at i = " << i << ": 1 = " << array_1[i] << ", 2 = " << array_2[i] << endl;
      // 	diff++;
      // } else {
      // 	same++;
      // }
    }
    cout << "same: " << same << endl;
    cout << "diff: " << diff << endl;
    if(max_diff_defined)
      cout << "max diff: " << max_diff << endl;
    else
      cout << "No max diff" << endl;
  }
}
