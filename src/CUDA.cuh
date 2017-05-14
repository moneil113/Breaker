#ifndef _CUDA_CUH_
#define _CUDA_CUH_
#include "ParticleSim.h"

void initCuda(SimulationData *data);

// void kernel(float *data, int size);

void runKernel(float *position_d, int size);

void copyParticles(SimulationData *data, float *newPosition_h, float *newVelocity_h, int newParticles);

void calcGrid(SimulationData *data);

void sortGrid(SimulationData *data);

void collideParticles(SimulationData *data);

void interactBoundaries(SimulationData *data /* boundary structure? */);

void applyForces(SimulationData *data, float dt);

void copyDeviceToHost(void *data, void *data_d, int size);

#endif
