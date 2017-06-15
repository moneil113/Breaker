#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

bool checkCudaDevice() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cout << "CUDA error on device query: " << cudaGetErrorString(error) << '\n';
    }

    if (deviceCount == 0) {
        std::cout << "No avaialable CUDA device: " << cudaGetErrorString(error) << '\n';
        return false;
    }

    cudaSetDevice(0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "CUDA device: " << deviceProp.name << '\n';

    printf("CUDA compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);

    return true;
}
