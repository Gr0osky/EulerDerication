#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__device__ double factorial(int n) {
    double result = 1.0;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void computeEuler(double *d_e_approx, int terms) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0.0;
    
    for (int i = idx; i < terms; i += gridDim.x * blockDim.x) {
        sum += 1.0 / factorial(i);
    }
    
    __shared__ double block_sum;
    if (threadIdx.x == 0) block_sum = 0.0;
    __syncthreads();
    
    atomicAddDouble(&block_sum, sum);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAddDouble(d_e_approx, block_sum);
    }
}

int main() {
    double h_e_approx = 0.0;
    double *d_e_approx;
    cudaMalloc((void **)&d_e_approx, sizeof(double));
    cudaMemcpy(d_e_approx, &h_e_approx, sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = 32;
    int terms = 100;

    auto start = std::chrono::high_resolution_clock::now();
    computeEuler<<<blocks, threads>>>(d_e_approx, terms);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&h_e_approx, d_e_approx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_e_approx);

    std::chrono::duration<double, std::milli> duration = end - start;
    
    printf("Approximated value of e (using CUDA high precision): %.100f\n", h_e_approx);
    printf("Execution Time (CUDA Advanced): %f ms\n", duration.count());
    
    return 0;
}
