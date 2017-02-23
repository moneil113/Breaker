#include "Particle.h"

#include <iostream>

#include "GLSL.h"
#include "MatrixStack.h"
#include "Program.h"
#include "Texture.h"

using namespace std;
using namespace Eigen;

Particle::Particle(int index, vector<float> &posBuf, vector<float> &colBuf) :
	x(&posBuf[3*index]),
    color(&colBuf[3*index])

{
}

Particle::~Particle() {
}

float randFloat(float l, float h) {
    float r = rand() / (float)RAND_MAX;
    return (1.0f - r) * l + r * h;
}

void Particle::rebirth(int spawnType) {
    float t = randFloat(0.0f, 1.0f);
    float z = randFloat(2.5f, 3.5f);
    
    if (spawnType == 0) {
        x << 5.0f + t, 0.0f + t, z;
        v << -7.0f, 7.0f, 0.0f;
    }
    else if (spawnType == 1) {
        x << 6.0f, 0.0f + t, z;
        v << -10.0f, 0.0f, 0.0f;
    }
    else if (spawnType == 2) {
        x << 2.5f + t, 4.0f, z;
        v << 0.0f, -10.0f, 0.0f;
    }
    else {
        x << 2.5f + t, 0.0f, z;
        v << 0.0f, 10.0f, 0.0f;
    }
}

float Particle::distance2(std::shared_ptr<Particle> other) {
    float a = x.x() - other->x.x();
    float b = x.y() - other->x.y();
    float c = x.z() - other->x.z();
    return a * a + b * b + c * c;
}

float Particle::distance(std::shared_ptr<Particle> other) {
    return sqrtf(distance2(other));
}
