#include <iostream>

#define kDataLen 1000

__global__ void child(float *z, int i) {
    z[i] += 1.f;
}

__global__ void parent(float* x, float* y, float *z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  z[i] += y[i] + x[i];
  child<<<1, 1>>>(z, i);
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

    bool passed = true;
    // Print the results.
    for (int i = 0; i < kDataLen; ++i) {
      std::cout << "y[" << i << "] = " << host_c[i];
      float expected = host_a[i] + host_b[i] + 1;
      std::cout << "    Expected: " << expected << std::endl;
      if(host_c[i] != expected) {
	passed = false;
      }
    }
    if(passed) 
      std::cout << "PASSED" << std::endl;
    else
      std::cout << "FAILED" << std::endl;
    
    cudaDeviceReset();
    return 0;
}
