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
#include <thread>

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

typedef std::vector<std::shared_ptr<Particle>> Bucket_t;

class Hash {
public:
    Hash(float bucketSize);
    virtual ~Hash();
    void add(std::shared_ptr<Particle> p);
    void clear();
    void step();
    
private:
    float bucketSize;
    // 3D grid of buckets
    std::vector<std::vector<std::vector<Bucket_t>>> buckets;
    // vector of buckets
    std::vector<Bucket_t *> groupedBuckets;
    std::vector<std::thread> threadHandles;
    
    Triplet hash(std::shared_ptr<Particle> p);
    Triplet hashMinus(std::shared_ptr<Particle> p);
    Triplet hashPlus(std::shared_ptr<Particle> p);
    void stepBucket(Triplet t);
    void threadStep(int threadId);
    
    // testing function to color particles based on buckets
    void colorBuckets();
};

#endif /* Hash_h */
