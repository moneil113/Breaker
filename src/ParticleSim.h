//
//  ParticleSim.h
//  Breaker
//
//  Created by Matthew O'Neil on 6/8/16.
//
//

#ifndef ParticleSim_h
#define ParticleSim_h

#include <memory>
#include <vector>
#include <thread>
#include <fstream>
#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>

#define SPAWN_RATE 10
#define NUM_SPAWNS 4

#define VELOCITY_COLOR 10.0f


class MatrixStack;
class Program;
class Particle;
class Hash;

class ParticleSim
{
public:
    ParticleSim(int n, float bucketSize);
    virtual ~ParticleSim();
    void stepParticles(float t, float dt, Eigen::Vector3f &g, const bool *keyToggles);
    void draw(std::shared_ptr<Program> prog);
    void reInit();
    void bakeFrame();
    
private:
    void emptyBuckets();
    void spawnParticles();
    
    void threadStepParticles(int threadId, float dt, Eigen::Vector3f g, const bool *keyToggles);
    
    Eigen::Vector3f lerp(float t);
    void assignColor(std::shared_ptr<Particle> p);
    
    bool spawningParticles;
    int activeParticles;
    int spawnType;
    
    int n;
    float m; // uniform particle mass
    std::vector<float> posBuf;
    std::vector<float> colBuf;
    GLuint posBufID;
    GLuint colBufID;
    
    std::vector<std::shared_ptr<Particle>> particles;
    
    // Cells/Buckets
    std::shared_ptr<Hash> grid;
    
    Eigen::Vector3f slowColor;
    Eigen::Vector3f fastColor;
    
    std::vector<std::thread> threadHandles;
    
    // Baked simulation data
    std::ofstream bakedFile;
    int frame;
};

#endif /* ParticleSim_h */
