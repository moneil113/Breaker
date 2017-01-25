//
//  Hash.h
//  Breaker
//
//  Created by Matthew O'Neil on 9/11/16.
//
//

#ifndef Hash_h
#define Hash_h

#include <vector>

#define EPSILON 0.0081f
// for sticky fluid: elasticity = 0.1
// for medium fluid: elasticity = 0.125
// for bouncy fluid: elasticity = 0.25
#define ELASTICITY 0.125f
// for viscous fluid: viscosity gain = 0.2
// for medium fluid: viscosity gain = 0.5
// for nonviscous fluid: viscosity gain = 0.8
#define VISCOSITY_GAIN 0.5f

#define RADIUS 0.08f

#define BUCKET_OVERLAP 0.1f

class Particle;
struct Triplet;

class Hash {
public:
    Hash(float bucketSize);
    virtual ~Hash();
    void add(std::shared_ptr<Particle> p);
    void clear();
    void step();
    
private:
    float bucketSize;
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<Particle>>>>> buckets;
    
    Triplet hash(std::shared_ptr<Particle> p);
    Triplet hashMinus(std::shared_ptr<Particle> p);
    Triplet hashPlus(std::shared_ptr<Particle> p);
    void stepBucket(Triplet t);
};

#endif /* Hash_h */
