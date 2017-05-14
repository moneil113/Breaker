#include "ParticleSim.h"

void initCuda(SimulationData *data);

// void kernel(float *data, int size);

void runKernel(float *position_d, int size);

void copyParticles(SimulationData *data, float *newPosition_h, float *newVelocity_h, int newParticles);

// particleHash_d is an int2 array of the form (grid cell, particle id)
void calcGrid(SimulationData *data);

void sortGrid(SimulationData *data);

void collideParticles(float *position_d, float *velocity_d, int size);

void interactBoundaries(float *position_d, float *velocity_d, int size /* boundary structure? */);

void applyForces(float *position_d, float *velocity_d, int size, float dt);

void copyDeviceToHost(void *data, void *data_d, int size);
