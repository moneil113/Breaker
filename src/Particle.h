#pragma once
#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <memory>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>

class MatrixStack;
class Program;
class Texture;

class Particle
{
public:
	
    Particle(int index, std::vector<float> &posBuf, std::vector<float> &colBuf);
	virtual ~Particle();
	void rebirth(int spawnType);
    float distance2(std::shared_ptr<Particle> other); // returns the distance squared between two particles
    float distance(std::shared_ptr<Particle> other); // returns the distance between two particles

    Eigen::Map<Eigen::Vector3f> x; // position (mapped to a location in posBuf)
    Eigen::Vector3f v;             // velocity
    
    Eigen::Map<Eigen::Vector3f> color; // color (mapped to a location in colBuf)
};

#endif
