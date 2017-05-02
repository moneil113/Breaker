#include "ParticleSim.h"
#include "CUDA.cuh"
#include "GLSL.h"
#include <iostream>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

ParticleSim::ParticleSim(int size) :
    size(size)
{
    data = new float[3 * size];
}

ParticleSim::~ParticleSim() {
}

void ParticleSim::init() {
std::cout << "init" << '\n';
test();
    createVBO(cudaGraphicsMapFlagsWriteDiscard);
    float *dptr;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t bytes;
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &bytes, cuda_vbo_resource));
std::cout << "init post cuda" << '\n';

    initCuda(&data, &dptr, size);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
std::cout << "init done" << '\n';
}

float randomFloat(float l, float h) {
    float r = rand() / (float)RAND_MAX;
    return (1.0f - r) * l + r * h;
}

void ParticleSim::test() {
    // data = new Vector3f[size];

    for (int i = 0; i < size; i++) {
        data[3 * i] = 0.1f * (i % 10);
        data[3 * i + 1] = 0.1f * (i / 10);
        data[3 * i + 2] = 1.0f;
    }
    cout << "allocated\n";
}

void ParticleSim::createVBO(unsigned int vbo_res_flags) {
std::cout << "createVBO" << '\n';
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, vbo_res_flags));
std::cout << "createVBO done" << '\n';
}

void ParticleSim::step() {
// std::cout << "step" << '\n';
    float *dptr;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t bytes;
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &bytes, *(&cuda_vbo_resource)));
// std::cout << "step post cuda" << '\n';
    // launch kernel
    runKernel(dptr, size);
// std::cout << "step post kernel" << '\n';
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
// std::cout << "step done" << '\n';
}

void ParticleSim::print() {
std::cout << "print" << '\n';
    float *dptr;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t bytes;
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &bytes, cuda_vbo_resource));
std::cout << "print before copy" << '\n';
    copyDeviceToHost(data, dptr, size);
std::cout << "print after copy" << '\n';

    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    if (data) {
        for (int i = 0; i < size; i++) {
            // Vector3f v = data[i];
            cout << data[3 * i] << " " << data[3 * i + 1] << " " << data[3 * i + 2] << endl;
        }
    }
    else {
        cout << data << endl;
    }
std::cout << "print done" << '\n';
}

void ParticleSim::draw(std::shared_ptr<Program> prog) {
// std::cout << "render" << '\n';
    // Enable and bind position array
    glEnableVertexAttribArray(prog->getAttribute("aPos"));
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // glBufferData(GL_ARRAY_BUFFER, 3 * size * sizeof(float), 0, GL_STATIC_DRAW);
    glVertexAttribPointer(prog->getAttribute("aPos"), 3, GL_FLOAT, GL_FALSE, 0, 0);
    // glVertexPointer(3, GL_FLOAT, 0, 0);

    // Draw
    // glEnableClientState(GL_VERTEX_ARRAY);
    // glColor3f(1.0, 1.0, 1.0);
    glDrawArrays(GL_POINTS, 0, size);
    // glDisableClientState(GL_VERTEX_ARRAY);

    // Disable and unbind
    glDisableVertexAttribArray(prog->getAttribute("aPos"));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
