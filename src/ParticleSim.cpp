#include "ParticleSim.h"
#include "CUDA.cuh"
#include "GLSL.h"
#include <iostream>
#include <iomanip>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

ParticleSim::ParticleSim(int size) :
    size(size)
{
    // data = new float[3 * size];
}

ParticleSim::~ParticleSim() {
}

void ParticleSim::init() {
    // data = {};
    createVBO(cudaGraphicsMapFlagsWriteDiscard);
    data.position_d = mapGLBuffer();
    data.activeParticles = 0;
    data.n = size;
    // data.minX = -3;
    // data.maxX = -3;
    // data.minY = 0;
    // data.maxY = 6;
    // data.minZ = -3;
    // data.maxZ = -3;
    data.gravity[0] = 0;
    data.gravity[1] = -9.8;
    data.gravity[2] = 0;

    initCuda(&data);
    cout << "(" << data.minX << ", " << data.minY << ", " << data.minZ << ") to ";
    cout << "(" << data.maxX << ", " << data.maxY << ", " << data.maxZ << ")\n";
    cout << data.totalCells << endl;

    unmapGLBuffer();
}

float randomFloat(float l, float h) {
    float r = rand() / (float)RAND_MAX;
    return (1.0f - r) * l + r * h;
}

void ParticleSim::spawnParticles() {
    float newPos[10 * 3];
    float newVel[10 * 3];
    for (size_t i = 0; i < 10; i++) {
        newPos[3 * i] = randomFloat(-0.2, 0.2);
        newPos[3 * i + 1] = randomFloat(2.8, 3.2);
        newPos[3 * i + 2] = randomFloat(-0.2, 0.2);

        newVel[3 * i] = randomFloat(-1, 1);
        newVel[3 * i + 1] = randomFloat(-1, 1);
        newVel[3 * i + 2] = randomFloat(-1, 1);
    }
    // copyParticles(&data, newPos, newVel, 10);
    // newPos[0] = -0.2;
    // newPos[1] = 3;
    // newPos[2] = 0;
    // newVel[0] = 1;
    // newVel[1] = 0;
    // newVel[2] = 0;
    //
    // newPos[3] = 0.2;
    // newPos[4] = 3;
    // newPos[5] = 0;
    // newVel[3] = -1;
    // newVel[4] = 0;
    // newVel[5] = 0;

    copyParticles(&data, newPos, newVel, 10);
}

void ParticleSim::createVBO(unsigned int vbo_res_flags) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, vbo_res_flags));
}

void ParticleSim::step(float dt) {
    data.position_d = mapGLBuffer();

    // TODO create sinks and sources for particles
    if (data.activeParticles < data.n) {
        spawnParticles();
    }

    // cout << "-----\n";

    calcGrid(&data);

    sortGrid(&data);

    collideParticles(&data);

    interactBoundaries(&data);

    applyForces(&data, dt);

    unmapGLBuffer();
}

void ParticleSim::print() {
    int count = data.activeParticles;
    float *localPos = new float[count * 3];
    copyDeviceToHost(localPos, data.position_d, count * 3 * sizeof(float));

    float *localVel = new float[count * 3];
    copyDeviceToHost(localVel, data.velocity_d, count * 3 * sizeof(float));

    int *localHash = new int[count];
    copyDeviceToHost(localHash, data.particleHash_d, count * sizeof(int));
    cout << "--------------\n";
    for (int i = 0; i < count; i++) {
        cout << "Particle " << i << ":\n";
        cout << " Position: (" << localPos[3 * i] << ", " << localPos[3 * i + 1] << ", " << localPos[3 * i + 2] << ")";
        cout << " Velocity: (" << localVel[3 * i] << ", " << localVel[3 * i + 1] << ", " << localVel[3 * i + 2] << ")";
        cout << " Cell: " << localHash[i] << endl;
    }
    cout << "--------------\n";

    delete[] localPos;
    delete[] localVel;
    delete[] localHash;
}

void ParticleSim::draw(std::shared_ptr<Program> prog) {
    // Enable and bind position array
    glEnableVertexAttribArray(prog->getAttribute("aPos"));
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(prog->getAttribute("aPos"), 3, GL_FLOAT, GL_FALSE, 0, 0);

    // Draw
    glDrawArrays(GL_POINTS, 0, data.activeParticles);

    // Disable and unbind
    glDisableVertexAttribArray(prog->getAttribute("aPos"));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

float *ParticleSim::mapGLBuffer() {
    float *dptr;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t bytes;
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &bytes, *(&cuda_vbo_resource)));
    return dptr;
}

void ParticleSim::unmapGLBuffer() {
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit(1);
}
