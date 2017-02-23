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

    // TODO this is another target for parallelization
    for (int i = 0; i < n; i++) {
        cout << i << endl;
        particles[i] = make_shared<Particle>(i, posBuf, colBuf);
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
    
    // Create a file to save baked simulation data
//    bakedFile.open("bakedData.brk");
//    bakedFile << n << endl;
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
    // Solver is adapted from Jos Stam's paper "Stable Fluids" and Intel's "Fluid Simulation for Video Games"
    // http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/ns.pdf
    // https://software.intel.com/en-us/articles/fluid-simulation-for-video-games-part-15
    
    if (spawningParticles) {
        spawnParticles();
    }

    // TODO this is the next target for parallelization
    
    

//    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
//        Vector3f newG = g;
//        threadHandles[threadId] = thread( [this, threadId, dt, newG, keyToggles]
//                                            { threadStepParticles(threadId, dt, newG, keyToggles); }
//                                         );
//    }
    
    for (int i = 0; i < activeParticles; i++) {
//        threadHandles[threadId].join();
        auto a = particles[i];
        Vector3f f = g * m;
        a->v += (dt/m) * f;
        a->x += dt * a->v;
        
        // Particle interactions with boundaries
        if (keyToggles[(unsigned) '1']) {
            if (keyToggles[(unsigned) '2']) {
                // bunch everything on the right hand side
                if (a->x.y() < 0) {
                    a->x.y() = randomFloat(0.0f, 0.01f);
                    a->v.y() = abs(a->v.y()) * ELASTICITY;
                }
                
                if (a->x.x() < 4) {
                    a->x.x() = randomFloat(4.0f, 4.01f);
                    a->v.x() = abs(a->v.x()) * ELASTICITY;
                }
                else if (a->x.x() > 6) {
                    a->x.x() = randomFloat(5.99f, 6.0f);
                    a->v.x() = abs(a->v.x()) * -ELASTICITY;
                }
            }
            else {
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
            }
        }
        else if (keyToggles[(unsigned) '3']) {
            if (a->x.y() < 0) {
                a->x.y() = randomFloat(0.0f, 0.01f);
                a->v.y() = abs(a->v.y()) * ELASTICITY;
            }
            else if (a->x.y() > 2) {
                a->x.y() = randomFloat(1.99f, 2.0f);
                a->v.y() = abs(a->v.y()) * -ELASTICITY;
            }
            
            if (a->x.x() < 0) {
                a->x.x() = randomFloat(0.0f, 0.01f);
                a->v.x() = abs(a->v.x()) * ELASTICITY;
            }
            else if (a->x.x() > 6) {
                a->x.x() = randomFloat(5.99f, 6.0f);
                a->v.x() = abs(a->v.x()) * -ELASTICITY;
            }
        }
        
        // Uncomment to impose boundaries in the z-axis
        if (a->x.z() < 0) {
            a->x.z() = randomFloat(0.0f, 0.01f);
            a->v.z() = abs(a->v.z()) * ELASTICITY;
        }
        else if (a->x.z() > 6) {
            a->x.z() = randomFloat(5.99f, 6.0f);
            a->v.z() = abs(a->v.z()) * -ELASTICITY;
        }
        assignColor(a);
        grid->add(a);
    }
    
    grid->step();
    grid->clear();
    
    frame++;
}

void ParticleSim::threadStepParticles(int threadId, float dt, Eigen::Vector3f g, const bool *keyToggles) {
    int span = (activeParticles + NUM_THREADS - 1) / NUM_THREADS;
    int start = threadId * span;
    int end = start + span;
    
    for (int i = start; i < end && i < activeParticles; i++) {
        auto a = particles[i];
        Vector3f f = g * m;
        a->v += (dt/m) * f;
        a->x += dt * a->v;
        
        // Particle interactions with boundaries
        if (keyToggles[(unsigned) '1']) {
            if (keyToggles[(unsigned) '2']) {
                // bunch everything on the right hand side
                if (a->x.y() < 0) {
                    a->x.y() = randomFloat(0.0f, 0.01f);
                    a->v.y() = abs(a->v.y()) * ELASTICITY;
                }
                
                if (a->x.x() < 4) {
                    a->x.x() = randomFloat(4.0f, 4.01f);
                    a->v.x() = abs(a->v.x()) * ELASTICITY;
                }
                else if (a->x.x() > 6) {
                    a->x.x() = randomFloat(5.99f, 6.0f);
                    a->v.x() = abs(a->v.x()) * -ELASTICITY;
                }
            }
            else {
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
            }
        }
        else if (keyToggles[(unsigned) '3']) {
            if (a->x.y() < 0) {
                a->x.y() = randomFloat(0.0f, 0.01f);
                a->v.y() = abs(a->v.y()) * ELASTICITY;
            }
            else if (a->x.y() > 2) {
                a->x.y() = randomFloat(1.99f, 2.0f);
                a->v.y() = abs(a->v.y()) * -ELASTICITY;
            }
            
            if (a->x.x() < 0) {
                a->x.x() = randomFloat(0.0f, 0.01f);
                a->v.x() = abs(a->v.x()) * ELASTICITY;
            }
            else if (a->x.x() > 6) {
                a->x.x() = randomFloat(5.99f, 6.0f);
                a->v.x() = abs(a->v.x()) * -ELASTICITY;
            }
        }
        
        // Uncomment to impose boundaries in the z-axis
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
