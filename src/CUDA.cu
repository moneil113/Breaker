#include "ParticleSim.h"
#define BLOCK_SIZE 16

void initCuda(float **data, float **data_g, int size) {
    // Vector3f *data_g;

    // cudaMalloc((void **)&data_g, sizeof(Vector3f) * size * size);

    cudaMemcpy(*data_g, *data, 3 * sizeof(float) * size, cudaMemcpyHostToDevice);
}

__global__ void kernel(float *data, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // data[row * width + col] += (Vector3f) {1.f, 2.f, 3.f};
    // data[row * width + col].x += 1.f;
    // data[row * width + col].y += 2.f;
    // data[row * width + col].z += 3.f;
    // data[id].x += 0.01f;
    // data[id].y += 0.02f;
    // data[id].z += 0.03f;
    if (id < size) {
        data[3 * id] += 0.01f * (id % 10);
        data[3 * id + 1] += 0.01f * (id / 10);
        // data[3 * id + 2] = 1.0f;
        // data[3 * id] += 0.1f;
        // data[3 * id + 1] += 0.1f;
        // data[3 * id + 2] += 0.1f;
    }
}



void runKernel(float *data_g, int size) {
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    kernel<<<dimBlock, dimGrid>>>(data_g, size);
}

void copyDeviceToHost(float *data, float *data_g, int size) {
    cudaMemcpy(data, data_g, 3 * sizeof(float) * size, cudaMemcpyDeviceToHost);
}
