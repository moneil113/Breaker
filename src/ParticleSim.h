#pragma once
#ifndef _PARTICLE_SIM_H_
#define _PARTICLE_SIM_H_

#include <vector>
#include <memory>

#include <GL/glew.h>

#include "Program.h"

// struct Vector3f {
//     float x;
//     float y;
//     float z;
// };

struct SimulationData {
    // I'd like to have the bounding box update at each time step,
    // but for now we'll just restrict the simulation space to a box
    float minX, maxX;
    float minY, maxY;
    float minZ, maxZ;
    float *position_d;
    float *velocity_d;
    int *particleHash_d;
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

public:
    ParticleSim (int size);
    virtual ~ParticleSim ();
    void draw(std::shared_ptr<Program> prog);
    void init();
    void step();
    void print();
};

#endif
