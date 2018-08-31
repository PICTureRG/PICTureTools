
__global__ void child(float *z) {
    z[threadIdx.x] += 1;
}

