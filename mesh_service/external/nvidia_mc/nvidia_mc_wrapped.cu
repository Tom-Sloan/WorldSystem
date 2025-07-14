// Wrapper for NVIDIA Marching Cubes kernels
// This file includes the NVIDIA implementation wrapped in our namespace

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

// Our simplified helper functions
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

namespace nvidia_mc {

// Include the tables
#include "tables.h"

// Include the original kernel code
#include "marchingCubes_kernel.cu"

} // namespace nvidia_mc