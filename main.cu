// IDEA GIVEN BY - ADHIRAJ1336 aka ADHIRAJ MISHTRA

// IMPORTS
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

// DEFINITIONS
#define TERMS 1000  // Number of terms for Taylor Series Approximation
#define THREADS_PER_BLOCK 256  // Threads per block in the kernel
#define MAX_BLOCKS 256  // Max blocks for execution
#define PRECISION 100  // Precision for e calculation (100 decimal places)
#define DIGITS_PER_THREAD 4  // Each thread handles 4 digits of precision

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error (%d): %s in %s at line %d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// High-precision representation of e using an array
typedef struct {
    int digits[PRECISION + 1]; 
} HighPrecision;

// Custom atomicAdd function for double
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// Custom power function using binary exponentiation
__device__ double custom_pow(double base, int exp) {
    double result = 1.0;
    while (exp > 0) {
        if (exp % 2 == 1) result *= base;
        base *= base;
        exp /= 2;
    }
    return result;
}

// Warp-level reduction
__device__ double warp_reduce(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ double block_reduce(double val) {
    __shared__ double shared[THREADS_PER_BLOCK];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce(val);
    return val;
}

// Kernel to compute e using Taylor series expansion
__global__ void compute_e_kernel(double *d_e_approx, int terms) {
    __shared__ double shared_sum;  

    if (threadIdx.x == 0) shared_sum = 0.0;  
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local_sum = 0.0;

    if (idx < terms) {
        double factorial = 1.0;
        for (int i = 1; i <= idx; i++) {
            factorial *= i;
        }
        local_sum = 1.0 / factorial;
    }

    double block_sum = block_reduce(local_sum);
    if (threadIdx.x == 0) {
        atomicAddDouble(d_e_approx, block_sum);
    }
}

// Function to compute Euler's number (e) using CUDA
HighPrecision compute_e_cuda_advanced(int terms) {
    double *d_e_approx;
    double h_e_approx = 0.0;

    CUDA_CHECK(cudaMalloc(&d_e_approx, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_e_approx, &h_e_approx, sizeof(double), cudaMemcpyHostToDevice));

    int num_blocks = (terms + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    num_blocks = num_blocks > MAX_BLOCKS ? MAX_BLOCKS : num_blocks;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    compute_e_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(d_e_approx, terms);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&h_e_approx, d_e_approx, sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_e_approx));
    CUDA_CHECK(cudaStreamDestroy(stream));

    HighPrecision e_high_precision;
    double integer_part = floor(h_e_approx);
    double fractional_part = h_e_approx - integer_part;

    e_high_precision.digits[0] = (int)integer_part;
    for (int i = 1; i <= PRECISION; i++) {
        fractional_part *= 10;
        e_high_precision.digits[i] = (int)fractional_part;
        fractional_part -= (int)fractional_part;
    }

    return e_high_precision;
}

int main() {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start, 0);  
    HighPrecision e_value = compute_e_cuda_advanced(TERMS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Approximated value of e (using Taylor series and CUDA optimizations): %d.", e_value.digits[0]);
    for (int i = 1; i <= PRECISION; i++) {
        printf("%d", e_value.digits[i]);
    }
    printf("\nExecution Time (CUDA Advanced): %f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}