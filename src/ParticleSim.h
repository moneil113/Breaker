#pragma once
#ifndef PARTICLE_SIM_H
#define PARTICLE_SIM_H

#include <vector>
#include <memory>

#include <GL/glew.h>

#include "Program.h"

struct SimulationData {
    // I'd like to have the bounding box update at each time step,
    // but for now we'll just restrict the simulation space to a box
    float minX, maxX;
    float minY, maxY;
    float minZ, maxZ;
    float gravity[3];
    int numCells[3]; // number of cells in x, y, z
    int totalCells; // total numbers of cells in flattened array
    float *position_d;
    float *velocity_d;
    float *nextVelocity_d;
    // particleHash_d gives the cell that a particle is in
    int *particleHash_d;
    // cellIndices_d is an int2 array of the form (start index, end index)
    // has width * height * depth entries, one for each cell
    int *cellIndices_d;
    int n;
    int activeParticles;
};

class ParticleSim {
private:
    int size; // number of particles to simulate
    SimulationData data;

    // GL interop values. Vertex buffer object for particle positions
    GLuint vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;

    // Device pointers
    float *velocity_d;
    int *particleHash_d;

    void createVBO(unsigned int vbo_res_flags);
    float *mapGLBuffer();
    void unmapGLBuffer();
    void spawnParticles();

    int spawnType;

public:
    ParticleSim (int size);
    virtual ~ParticleSim ();
    void draw(std::shared_ptr<Program> prog);
    void init();
    void step(float dt);
    void restart();
    void print();

    void freeze();
    void toggleGravity();
};

#endif
