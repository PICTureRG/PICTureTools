#include "child.cu"

#define kDataLen 1000
__global__ void parent(float* x, float* y, float *z) {
  z[threadIdx.x] += y[threadIdx.x] + x[threadIdx.x];
  child<<<1, kDataLen>>>(z);
}

