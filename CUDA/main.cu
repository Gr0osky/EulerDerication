#include <stdio.h>
#include <cuda.h>
#include <mpfr.h>

#define N 200  // Number of terms in the Taylor series (enough for 100 decimal places)
#define BLOCK_SIZE 256
#define PRECISION 332  // MPFR precision in bits (log2(10) * 100 ≈ 332 bits for 100 decimal places)

// CUDA error checking macro
#define CHECK_CUDA_CALL(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            fflush(stdout); \
            return -1; \
        } \
    }

// CUDA kernel to compute terms of the Taylor series
__global__ void compute_terms(double *terms, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        double term = 1.0;
        for (int i = 1; i <= idx; i++) {
            term /= (double)i;
        }
        terms[idx] = term;
    }
}

int main() {
    printf("Starting computation...\n");
    fflush(stdout);

    double h_terms[N] = {0};
    double *d_terms;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-compatible device found!\n");
        return -1;
}


    // Allocate memory on the device with error checking
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_terms, N * sizeof(double)));
    CHECK_CUDA_CALL(cudaMemset(d_terms, 0, N * sizeof(double))); // Ensure memory is initialized
    
    printf("Memory allocated on GPU.\n");
    fflush(stdout);

    // Define the grid and block dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch the CUDA kernel
    compute_terms<<<numBlocks, BLOCK_SIZE>>>(d_terms, N);
    CHECK_CUDA_CALL(cudaPeekAtLastError());  // Check for kernel launch errors
    CHECK_CUDA_CALL(cudaDeviceSynchronize());  // Ensure kernel execution completes
    
    printf("Kernel execution completed.\n");
    fflush(stdout);

    // Copy results back to host
    CHECK_CUDA_CALL(cudaMemcpy(h_terms, d_terms, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    printf("Data copied back to host.\n");
    fflush(stdout);

    // Initialize MPFR
    mpfr_t e, term;
    mpfr_init2(e, PRECISION);
    mpfr_init2(term, PRECISION);
    mpfr_set_d(e, 0.0, MPFR_RNDN);

    printf("MPFR initialized.\n");
    fflush(stdout);

    // Sum the terms using MPFR
    for (int i = 0; i < N; i++) {
        mpfr_set_d(term, h_terms[i], MPFR_RNDN);
        mpfr_add(e, e, term, MPFR_RNDN);
    }

    printf("Summation complete.\n");
    fflush(stdout);

    // Print the result to 100 decimal places
    printf("Calculated value of e to 100 decimal places:\n");
    fflush(stdout);
    mpfr_printf("%.100Rf\n", e);
    printf("\n");
    fflush(stdout);

    // Clean up
    mpfr_clear(e);
    mpfr_clear(term);
    CHECK_CUDA_CALL(cudaFree(d_terms));
    mpfr_free_cache();

    printf("Computation finished successfully.\n");
    fflush(stdout);

    return 0;
}