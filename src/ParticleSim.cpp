//
//  ParticleSim.cpp
//  Breaker
//
//  Created by Matthew O'Neil on 6/8/16.
//
//

#include "ParticleSim.h"
#include "Particle.h"
#include "Program.h"
#include "Hash.h"

#include "Util.h"

#include <iostream>
using namespace std;
using namespace Eigen;

ParticleSim::ParticleSim(int n, float bucketSize) :
    spawningParticles(true),
    activeParticles(0),
    spawnType(0),
    n(n),
    m(1.0f),
    frame(0)
{
    posBuf.resize(3*n);
    colBuf.resize(3*n);
    particles = vector<shared_ptr<Particle>>(n);

    grid = make_shared<Hash>(bucketSize);
    
    // Blue
    slowColor << 0.11f, 0.2f, 0.67f;
    fastColor << 0.43f, 0.65f, 1.0f;
    
    // Green
//    slowColor << 0.25f, 0.47f, 0.3f;
//    fastColor << 0.27f, 0.75f, 0.29f;
    
    // Purple
//    slowColor << 0.47f, 0.24f, 0.51f;
//    fastColor << 0.81f, 0.43f, 0.75f;

    // TODO this is another target for parallelization, but this only happens once
    //      and it doesn't take long
    for (int i = 0; i < n; i++) {
        cout << i << endl;
        particles[i] = make_shared<Particle>(i, posBuf, colBuf, 1.0f);
    }
    
    // Bind position buffer
    glGenBuffers(1, &posBufID);
    glBindBuffer(GL_ARRAY_BUFFER, posBufID);
    glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_STATIC_DRAW);
    
    // Bind color buffer
    glGenBuffers(1, &colBufID);
    glBindBuffer(GL_ARRAY_BUFFER, colBufID);
    glBufferData(GL_ARRAY_BUFFER, colBuf.size()*sizeof(float), &colBuf[0], GL_STATIC_DRAW);
    
    // Create thread handles for threading later on
    threadHandles = vector<thread>(NUM_THREADS);
}

ParticleSim::~ParticleSim() {
}

float randomFloat(float l, float h) {
    float r = rand() / (float)RAND_MAX;
    return (1.0f - r) * l + r * h;
}

void ParticleSim::spawnParticles() {
    for (int i = 0; i < SPAWN_RATE && activeParticles < n; i++) {
        particles[activeParticles]->rebirth(spawnType);
        activeParticles++;
    }
    if (activeParticles == n) {
        spawningParticles = false;
    }
}

void ParticleSim::stepParticles(float t, float dt, Eigen::Vector3f &g, const bool *keyToggles) {
    if (spawningParticles) {
        spawnParticles();
    }
    
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId] = thread( [this, threadId, dt, g, keyToggles]
                                            { threadStepParticles(threadId); }
                                         );
    }
    
    for (int i = 0; i < activeParticles; i++) {
        grid->add(particles[i]);
    }
    
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId].join();
    }
    
    grid->step();
    grid->clear();
    
    // after accumulating all forces acting on particles, apply forces
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId] = thread( [this, threadId, dt, g]
                                            { applyForces(threadId, dt, g); }
                                         );
    }
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId].join();
    }
    
    frame++;
}

void ParticleSim::threadStepParticles(int threadId) {
    int span = (activeParticles + NUM_THREADS - 1) / NUM_THREADS;
    int start = threadId * span;
    int end = start + span;
    
    for (int i = start; i < end && i < activeParticles; i++) {
        auto a = particles[i];
        
        // Velocity parallel to ground
        Vector3f vXZ = a->v;
        Vector3f vXY = vXZ;
        Vector3f vYZ = vXZ;
        vXZ.y() = 0.0f;
        vXY.z() = 0.0f;
        vYZ.x() = 0.0f;
        
        // Particle interactions with boundaries
        if (a->x.y() < 0) {
            a->x.y() = randomFloat(0.0f, 0.01f);
            a->v.y() = abs(a->v.y()) * ELASTICITY;
        }
        
        if (a->x.x() < 0) {
            a->x.x() = randomFloat(0.0f, 0.01f);
            a->v.x() = abs(a->v.x()) * ELASTICITY;
        }
        else if (a->x.x() > 6) {
            a->x.x() = randomFloat(5.99f, 6.0f);
            a->v.x() = abs(a->v.x()) * -ELASTICITY;
        }
        
        if (a->x.z() < 0) {
            a->x.z() = randomFloat(0.0f, 0.01f);
            a->v.z() = abs(a->v.z()) * ELASTICITY;
        }
        else if (a->x.z() > 6) {
            a->x.z() = randomFloat(5.99f, 6.0f);
            a->v.z() = abs(a->v.z()) * -ELASTICITY;
        }
        assignColor(a);
    }
}

void ParticleSim::applyForces(int threadId, float dt, const Eigen::Vector3f &g) {
    int span = (activeParticles + NUM_THREADS - 1) / NUM_THREADS;
    int start = threadId * span;
    int end = start + span;
    
    Vector3f f;
    
    for (int i = start; i < end && i < activeParticles; i++) {
        auto a = particles[i];
        f = g * a->mass();
        a->addForce(f);
        if (a->x.y() <= 0.09) {
            // add friction force
            a->addForce(0.01f * m * 9.8 * -a->v.normalized());
        }
        a->applyForces(dt);
    }
}

Vector3f ParticleSim::lerp(float t) {
    if (t > 1) {
        t = 1;
    }
    else if (t < 0) {
        t = 0;
    }
    
    float r = slowColor.x() + (fastColor.x() - slowColor.x()) * t;
    float g = slowColor.y() + (fastColor.y() - slowColor.y()) * t;
    float b = slowColor.z() + (fastColor.z() - slowColor.z()) * t;
    
    return Vector3f(r, g, b);
}

void ParticleSim::assignColor(std::shared_ptr<Particle> p) {
    float mag = p->v.norm();
    
    p->color = lerp(mag / VELOCITY_COLOR);
}

void ParticleSim::bakeFrame() {
    stringstream s;
    s << "/tmp/breaker/breakerSim_" << frame << ".brk";
    bakedFile.open(s.str(), ios::out | ios::binary);

    bakedFile.write((const char *)&activeParticles, sizeof(int));
    for (int i = 0; i < activeParticles; i++) {
        auto a = particles[i];
        Vector3f x = a->x;
        float v = a->v.norm();
        bakedFile.write((const char *)&x, sizeof(Vector3f));
        bakedFile.write((const char *)&v, sizeof(float));
    }
    bakedFile.close();
}

void ParticleSim::draw(std::shared_ptr<Program> prog) {
    // Enable, bind, and send position array
    glEnableVertexAttribArray(prog->getAttribute("aPos"));
    glBindBuffer(GL_ARRAY_BUFFER, posBufID);
    glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_STATIC_DRAW);
    glVertexAttribPointer(prog->getAttribute("aPos"), 3, GL_FLOAT, GL_FALSE, 0, 0);
    
    // Enable and bind color array
    glEnableVertexAttribArray(prog->getAttribute("aCol"));
    glBindBuffer(GL_ARRAY_BUFFER, colBufID);
    glBufferData(GL_ARRAY_BUFFER, colBuf.size()*sizeof(float), &colBuf[0], GL_STATIC_DRAW);
    glVertexAttribPointer(prog->getAttribute("aCol"), 3, GL_FLOAT, GL_FALSE, 0, 0);
    
    // Draw
    glDrawArrays(GL_POINTS, 0, activeParticles);
    
    // Disable and unbind
    glDisableVertexAttribArray(prog->getAttribute("aPos"));
    glDisableVertexAttribArray(prog->getAttribute("aCol"));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSim::reInit() {
    spawningParticles = true;
    activeParticles = 0;
    spawnType = (spawnType + 1) % NUM_SPAWNS;
}
