// Implementation of device kernels and calling code

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
#define RADIUS 0.1f
#define CELL_SIZE (2 * RADIUS)
#define ELASTICITY 0.5f
#define VISCOSITY_GAIN 0.5f
#define DAMPING 0.1f

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

int3 numCells(float3 dimensions) {
    int x = (int) ceilf((dimensions.x) / CELL_SIZE);
    int y = (int) ceilf((dimensions.y) / CELL_SIZE);
    int z = (int) ceilf((dimensions.z) / CELL_SIZE);

    return make_int3(x, y, z);
}

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

    CUDA_CHECK_RETURN(cudaMalloc((void **)&hash_d, sizeof(int) * size));

    // Set up cell index array
    int *indices_d;
    int3 cells = numCells(make_float3(6, 6, 6));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&indices_d,
                                 sizeof(int2) * cells.x * cells.y * cells.z));

    data->velocity_d = vel_d;
    data->particleHash_d = hash_d;
    data->cellIndices_d = indices_d;

    data->minX = -3;
    data->maxX = 3;
    data->minY = 0;
    data->maxY = 6;
    data->minZ = -3;
    data->maxZ = 3;

    data->numCells[0] = cells.x;
    data->numCells[1] = cells.y;
    data->numCells[2] = cells.z;

    data->totalCells = cells.x * cells.y * cells.z;
}

void copyParticles(SimulationData *data,
                   float *newPosition_h,
                   float *newVelocity_h,
                   int newParticles) {
    int active = data->activeParticles;
    float *start = data->position_d + active * 3;
    CUDA_CHECK_RETURN(cudaMemcpy(start, newPosition_h,
                                 newParticles * sizeof(float3),
                                 cudaMemcpyHostToDevice));

    start = data->velocity_d + active * 3;
    CUDA_CHECK_RETURN(cudaMemcpy(start, newVelocity_h,
                                 newParticles * sizeof(float3),
                                 cudaMemcpyHostToDevice));

    data->activeParticles += newParticles;
}

// Hashing
__device__ int3 getCell(float3 position, float3 min) {
    int x = (int) ((position.x - min.x) / CELL_SIZE);
    int y = (int) ((position.y - min.y) / CELL_SIZE);
    int z = (int) ((position.z - min.z) / CELL_SIZE);

    return make_int3(x, y, z);
}

__device__ int cellNumber(int3 cell, int3 numCells) {
    int width = numCells.x;
    int height = numCells.y;
    return cell.x + cell.y * width + cell.z * width * height;
}

__global__ void hashParticles(float3 *position_d,
                              int *particleHash_d,
                              float3 min,
                              int3 numCells,
                              int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }

    float3 pos = position_d[id];

    int3 cell = getCell(pos, min);
    int cellId = cellNumber(cell, numCells);
    particleHash_d[id] = cellId;
}

void calcGrid(SimulationData *data) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *pos_d = (float3 *)data->position_d;
    int *hash_d = data->particleHash_d;
    float3 min = make_float3(data->minX, data->minY, data->minZ);
    int3 numCells = make_int3(data->numCells[0],
                              data->numCells[1],
                              data->numCells[2]);

    hashParticles<<<dimBlock, dimGrid>>>(pos_d, hash_d, min, numCells, size);
}

// Sorting
struct comparePairs {
    __host__ __device__
    bool operator()(const int &a, const int &b) {
        return a < b;
    }
};

__global__ void resetCells(int2 *cellIndices_d, int numCells) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numCells) {
        return;
    }

    cellIndices_d[id] = make_int2(-1, -1);
}

__global__ void cellBounds(int *particleHash_d, int2 *cellIndices_d, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }

    // TODO use shared memory to look at adjacent positions
    int hash = particleHash_d[id];
    if (id != 0) {
        int hashMinus = particleHash_d[id - 1];
        if (hash != hashMinus) {
            cellIndices_d[hash].x = id; // start of new cell
            cellIndices_d[hashMinus].y = id - 1; // end of previous cell
        }
        if (id == size - 1) {
            cellIndices_d[hash].y = id; // final particle in final cell
        }
    }
    else {
        cellIndices_d[hash].x = id; // first particle of first cell
    }
}

void sortGrid(SimulationData *data) {
    int size = data->activeParticles;
    // We want to sort both the position_d and velocity_d arrays using the keys
    // stored in the particleHash_d array
    thrust::device_ptr<float3> pos_ptr((float3 *)data->position_d);
    thrust::device_ptr<float3> vel_ptr((float3 *)data->velocity_d);
    thrust::device_ptr<int> hash_ptr(data->particleHash_d);

    thrust::sort_by_key(hash_ptr, hash_ptr + size,
        thrust::make_zip_iterator(thrust::make_tuple(pos_ptr, vel_ptr)),
        comparePairs());

    // find start and end index of each cell
    int3 numCells = make_int3(data->numCells[0],
                              data->numCells[1],
                              data->numCells[2]);
    int totalCells = numCells.x * numCells.y * numCells.z;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 cellGrid((totalCells + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int *hash_d = data->particleHash_d;

    int2 *cell_d = (int2 *)data->cellIndices_d;
    resetCells<<<dimBlock, cellGrid>>>(cell_d, totalCells);

    cellBounds<<<dimBlock, dimGrid>>>(hash_d, cell_d, size);
}

// Collision
// collide a single particle with every particle in the given cell
// returns a float3 giving its change in velocity
__device__ float3 collideParticleCell(float3 pos,
                                      float3 vel,
                                      int particleId,
                                      float3 *position_d,
                                      float3 *velocity_d,
                                      int2 indices) {
    float3 deltaV = make_float3(0, 0, 0);
    if (indices.x < 0) {
        // no start, so cell is empty
        return deltaV;
    }

    float3 otherPos, otherVel;

    for (int i = indices.x; i <= indices.y; i++) {
        // don't want to collide particle with itself
        if (i != particleId) {
            otherPos = position_d[i];
            otherVel = velocity_d[i];
            // dx points away from the other particle
            float3 dx = pos - otherPos;
            float dist = sqrtf(dot(dx, dx));
            if (dist < RADIUS * 2) {
                // direction from this particle to other particle
                float3 dir = normalize(dx);
                // spring  force (Hooke's law, F = kX)
                deltaV += dir * VISCOSITY_GAIN * (RADIUS * 2 - dist);
                // damping force
                float3 dv = vel - otherVel;
                deltaV -= dir * DAMPING * dot(dv, dx);
            }
        }
    }
    return deltaV;
}

__global__ void collide(float3 *position_d,
                        float3 *velocity_d,
                        int2 *cellIndices_d,
                        int3 numCells,
                        float3 min,
                        int totalCells,
                        int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    float3 vel;

    if (id < size) {
        float3 pos = position_d[id];
        vel = velocity_d[id];
        int3 cell = getCell(pos, min);

        // TODO use shared memory to look at neighboring cells
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    int3 otherCell = cell + make_int3(x, y, z);
                    int otherCellId = cellNumber(otherCell, numCells);
                    if (otherCellId < totalCells && otherCellId >= 0) {
                        int2 indices = cellIndices_d[otherCellId];
                        vel += collideParticleCell(pos, vel, id,
                                                   position_d, velocity_d,
                                                   indices);
                    }
                }
            }
        }
    }

    __syncthreads();

    if (id < size) {
        velocity_d[id] = vel;
    }
}

void collideParticles(SimulationData *data) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *pos_d = (float3 *)data->position_d;
    float3 *vel_d = (float3 *)data->velocity_d;
    int2 *cell_d = (int2 *)data->cellIndices_d;
    int3 numCells = make_int3(data->numCells[0],
                              data->numCells[1],
                              data->numCells[2]);
    float3 min = make_float3(data->minX, data->minY, data->minZ);
    int totalCells = data->totalCells;

    collide<<<dimBlock, dimGrid>>>(pos_d, vel_d, cell_d,
                                   numCells, min, totalCells, size);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

// Interact with boundaries
__global__ void boundaries(float3 *position_d, float3 *velocity_d,
                           int3 min, int3 max, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }

    float jitter[10] = {0.00413, 0.00093, 0.00554, 0.00092, 0.00103,
        0.00425, 0.00980, 0.00744, 0.00959, 0.00299};

    float3 pos = position_d[id];
    float3 vel = velocity_d[id];

    if (pos.x < min.x) {
        pos.x = min.x + jitter[(id % 13 + id % 7) % 10];
        vel.x *= -ELASTICITY;
    }
    if (pos.x > max.x) {
        pos.x = max.x - jitter[(id % 13 + id % 7) % 10];
        vel.x *= -ELASTICITY;
    }
    if (pos.y < min.y) {
        pos.y = min.y + jitter[(id % 3 + id % 19) % 10];
        vel.y *= -ELASTICITY;
    }
    if (pos.y > max.y) {
        pos.y = max.y - jitter[(id % 3 + id % 19) % 10];
        vel.y *= -ELASTICITY;
    }
    if (pos.z < min.z) {
        pos.z = min.z + jitter[(id % 5 + id % 23) % 10];
        vel.z *= -ELASTICITY;
    }
    if (pos.z > max.z) {
        pos.z = max.z - jitter[(id % 5 + id % 23) % 10];
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
    int3 min = make_int3(data->minX, data->minY, data->minZ);
    int3 max = make_int3(data->maxX, data->maxY, data->maxZ);

    boundaries<<<dimBlock, dimGrid>>>(pos_d, vel_d, min, max, size);
}

// Apply forces
__global__ void integrate(float3 *position_d,
                          float3 *velocity_d,
                          float3 gravity,
                          int size,
                          float dt) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= size) {
        return;
    }
    float3 vel = velocity_d[id];
    vel += gravity * dt;
    velocity_d[id] = vel;
    position_d[id] += vel * dt;
}

void applyForces(SimulationData *data, float dt) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *pos_d = (float3 *)data->position_d;
    float3 *vel_d = (float3 *)data->velocity_d;

    float3 gravity = make_float3(data->gravity[0],
                                 data->gravity[1],
                                 data->gravity[2]);

    integrate<<<dimBlock, dimGrid>>>(pos_d, vel_d, gravity, size, dt);
}

void freezeParticles(SimulationData *data) {
    int size = data->activeParticles;
    dim3 dimBlock = dim3(BLOCK_SIZE);
    dim3 dimGrid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float3 *vel_d = (float3 *)data->velocity_d;

    zeroArray<<<dimBlock, dimGrid>>>(vel_d, size);
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
