#include "Particle.h"

#include <iostream>

#include "GLSL.h"
#include "MatrixStack.h"
#include "Program.h"

using namespace std;
using namespace Eigen;

Particle::Particle(int index, vector<float> &posBuf, vector<float> &colBuf, float mass) :
	x(&posBuf[3*index]),
    color(&colBuf[3*index])
{
    forces << 0.0f, 0.0f, 0.0f;
    m = mass;
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
        x << 2.5f + t, 0.1f, z;
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

void Particle::addForce(const Eigen::Vector3f &f) {
    mutex.lock();
    forces += f;
    mutex.unlock();
}

void Particle::applyForces(float dt) {
    // Find acceleration. f = ma => a  = f/m
    Vector3f a = forces / m;
    
    v += a * dt;
    x += v * dt;
    
    forces << 0.0f, 0.0f, 0.0f;
}
