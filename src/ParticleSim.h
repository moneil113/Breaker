#include <vector>
#include <memory>

#include <GL/glew.h>

#include "Program.h"

// struct Vector3f {
//     float x;
//     float y;
//     float z;
// };

class ParticleSim {
private:
    int size; // number of particles to simulate
    float *data;
    GLuint vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;

    void createVBO(unsigned int vbo_res_flags);

public:
    ParticleSim (int size);
    virtual ~ParticleSim ();
    void test();
    void draw(std::shared_ptr<Program> prog);
    void init();
    void step();
    void print();
};
