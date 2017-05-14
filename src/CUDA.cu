// #include "ParticleSim.h"
#include "CUDA.cuh"
#include <vector_types.h>
#include <vector_functions.h>
#include "VectorFuncs.cuh"
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 64
#define RADIUS 0.08f
#define CELL_SIZE (2 * RADIUS)
#define ELASTICITY 0.5f

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void zeroArray(float3 *data, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }
    data[id] = make_float3(0.0f, 0.0f, 0.0f);
}

void initCuda(SimulationData *data) {
    int size = data->n;

    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float *vel_d;
    int *hash_d;

    zeroArray<<<dimBlock, dimGrid>>>((float3 *)(data->position_d), size);

    CUDA_CHECK_RETURN(cudaMalloc((void **)&vel_d, sizeof(float3) * size));
    zeroArray<<<dimBlock, dimGrid>>>((float3 *)vel_d, size);

    CUDA_CHECK_RETURN(cudaMalloc((void **)&hash_d, sizeof(int2) * size));

    data->velocity_d = vel_d;
    data->particleHash_d = hash_d;

    data->minX = 3;
    data->maxX = -3;
    data->minY = 0;
    data->maxY = 6;
    data->minZ = -3;
    data->maxZ = 3;
}

void copyParticles(SimulationData *data, float *newPosition_h, float *newVelocity_h, int newParticles) {
    int active = data->activeParticles;
    float *start = data->position_d + active * 3;
    CUDA_CHECK_RETURN(cudaMemcpy(start, newPosition_h, newParticles * sizeof(float3), cudaMemcpyHostToDevice));

    start = data->velocity_d + active * 3;
    CUDA_CHECK_RETURN(cudaMemcpy(start, newVelocity_h, newParticles * sizeof(float3), cudaMemcpyHostToDevice));

    data->activeParticles += newParticles;
}

__global__ void kernel(float3 *data, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < size) {
        data[id].x += 0.01f * (id % 10);
        data[id].y += 0.01f * (id / 10);
    }
}

void runKernel(float *data_g, int size) {
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    kernel<<<dimBlock, dimGrid>>>((float3 *)data_g, size);
}

// Hashing functions
__device__ int3 getCell(float3 position) {
    int x = (int) (position.x / CELL_SIZE);
    int y = (int) (position.y / CELL_SIZE);
    int z = (int) (position.z / CELL_SIZE);

    return make_int3(x, y, z);
}

__device__ int cellNumber(int3 cell, float3 gridSize) {
    int width = (int) ceilf((gridSize.x) / CELL_SIZE);
    int height = (int) ceilf((gridSize.y) / CELL_SIZE);
    return cell.x + cell.y * width + cell.z * width * height;
}

__global__ void hashParticles(float3 *position_d, int2 *particleHash_d, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= size) {
        return;
    }

    int3 cell = getCell(position_d[id]);
    int cellId = cellNumber(cell, make_float3(6, 6, 6));
    particleHash_d[id] = make_int2(cellId, id);
    // position_d[id] = make_float3(cell.x, cell.y, cell.z);
}

void calcGrid(SimulationData *data) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *pos_d = (float3 *)data->position_d;
    int2 *hash_d = (int2 *)data->particleHash_d;

    hashParticles<<<dimBlock, dimGrid>>>(pos_d, hash_d, size);
}

// Sorting functions
struct comparePairs {
    __host__ __device__
    bool operator()(const int2 &a, const int2 &b) {
        return a.x < b.x;
    }
};

void sortGrid(SimulationData *data) {
    int size = data->activeParticles;
    // We want to sort both the position_d and velocity_d arrays using the keys
    // stored in the particleHash_d array
    thrust::device_ptr<float3> pos_ptr((float3 *)data->position_d);
    thrust::device_ptr<float3> vel_ptr((float3 *)data->velocity_d);
    thrust::device_ptr<int2> hash_ptr((int2 *)data->particleHash_d);

    thrust::sort_by_key(hash_ptr, hash_ptr + size,
        thrust::make_zip_iterator(thrust::make_tuple(pos_ptr, vel_ptr)),
        comparePairs());
}

// Collision
void collideParticles(SimulationData *data) {

}

// Interact with boundaries
void __global__ boundaries(float3 *position_d, float3 *velocity_d, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }

    float3 pos = position_d[id];
    float3 vel = velocity_d[id];

    if (pos.x < -3) {
        pos.x = -3;
        vel.x *= -ELASTICITY;
    }
    else if (pos.x > 3) {
        pos.x = 3;
        vel.x *= -ELASTICITY;
    }
    if (pos.y < 0) {
        pos.y = 0;
        vel.y *= -ELASTICITY;
    }
    else if (pos.y > 6) {
        pos.y = 6;
        vel.y *= -ELASTICITY;
    }
    if (pos.z < -3) {
        pos.z = -3;
        vel.z *= -ELASTICITY;
    }
    else if (pos.z > 3) {
        pos.z = 3;
        vel.z *= -ELASTICITY;
    }

    position_d[id] = pos;
    velocity_d[id] = vel;
}

void interactBoundaries(SimulationData *data) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *pos_d = (float3 *)data->position_d;
    float3 *vel_d = (float3 *)data->velocity_d;

    boundaries<<<dimBlock, dimGrid>>>(pos_d, vel_d, size);
}

// Apply forces
__global__ void integrate(float3 *position_d, float3 *velocity_d, int size, float dt) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }
    position_d[id] += velocity_d[id] * dt;
}

void applyForces(SimulationData *data, float dt) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *pos_d = (float3 *)data->position_d;
    float3 *vel_d = (float3 *)data->velocity_d;

    integrate<<<dimBlock, dimGrid>>>(pos_d, vel_d, size, dt);
}

// Utility functions
void copyDeviceToHost(void *data, void *data_d, int size) {
    CUDA_CHECK_RETURN(cudaMemcpy(data, data_d, size, cudaMemcpyDeviceToHost));
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;

    fprintf(stderr, "%s returned %s (%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}
