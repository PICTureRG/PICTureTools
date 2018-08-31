#include <iostream>

#define kDataLen 1000

__global__ void child(float *z) {
    z[threadIdx.x] += 1;
}

__global__ void parent(float* x, float* y, float *z) {
  z[threadIdx.x] += y[threadIdx.x] + x[threadIdx.x];

  if (threadIdx.x < 10) {
      return;
  }

  child<<<1, kDataLen>>>(z);
}

int main(int argc, char* argv[]) {
    float host_a[kDataLen];
    float host_b[kDataLen];
    float host_c[kDataLen];
  
    for (int i=0; i < kDataLen; i++) {
        host_a[i] = i;
        host_b[i] = 2*i;
    }
  
    // Copy input data to device.
    float* device_a;
    float* device_b;
    float* device_c;

    cudaMalloc(&device_a, kDataLen * sizeof(float));
    cudaMalloc(&device_b, kDataLen * sizeof(float));
    cudaMalloc(&device_c, kDataLen * sizeof(float));
  
    cudaMemcpy(device_a, host_a, kDataLen * sizeof(float), 
            cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, kDataLen * sizeof(float), 
            cudaMemcpyHostToDevice);
  
    // Launch the kernel.
    parent<<<5, kDataLen/5>>>(device_a, device_b, device_c);
  
    // Copy output data to host.
    cudaDeviceSynchronize();
    cudaMemcpy(host_c, device_c, kDataLen * sizeof(float), 
            cudaMemcpyDeviceToHost);
  
    // Print the results.
    for (int i = 0; i < kDataLen; ++i) {
      std::cout << "y[" << i << "] = " << host_c[i] << "\n";
    }
  
    cudaDeviceReset();
    return 0;
}
