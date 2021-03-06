//
//  Hash.cpp
//  Breaker
//
//  Created by Matthew O'Neil on 9/11/16.
//
//

#include "Hash.h"
#include "Particle.h"

#include "Util.h"

#include <iostream>
#include <sstream>

using namespace std;
using namespace Eigen;

typedef struct Triplet {
    int a;
    int b;
    int c;
} Triplet;

int addCount = 0;
int bucketCount = 0;

Hash::Hash(float bucketSize) {
    this->bucketSize = bucketSize;
    buckets = vector<vector<vector<Bucket_t>>>();
    
    threadHandles = vector<thread>(NUM_THREADS);
}

Hash::~Hash() {
}

Triplet Hash::hash(std::shared_ptr<Particle> p) {
    return {(int) (p->x.x() / bucketSize),
        (int) (p->x.y() / bucketSize),
        (int) (p->x.z() / bucketSize)};
}

Triplet Hash::hashMinus(std::shared_ptr<Particle> p) {
    return {(int) ((p->x.x() - bucketSize * BUCKET_OVERLAP) / bucketSize),
        (int) ((p->x.y() - bucketSize * BUCKET_OVERLAP) / bucketSize),
        (int) ((p->x.z() - bucketSize * BUCKET_OVERLAP) / bucketSize)};
}

Triplet Hash::hashPlus(std::shared_ptr<Particle> p) {
    return {(int) ((p->x.x() + bucketSize * BUCKET_OVERLAP) / bucketSize),
        (int) ((p->x.y() + bucketSize * BUCKET_OVERLAP) / bucketSize),
        (int) ((p->x.z() + bucketSize * BUCKET_OVERLAP) / bucketSize)};
}

struct BucketTriple {
    Bucket_t *p;
    Triplet t;
};

void Hash::add(std::shared_ptr<Particle> p) {
    Triplet result = hash(p);
#ifndef DEBUG
    Triplet resultMinus = hashMinus(p);
    Triplet resultPlus = hashPlus(p);
#endif
    
    if (buckets.size() <= result.a) {
        buckets.resize(result.a + 1);
    }
    if (buckets.at(result.a).size() <= result.b) {
        buckets.at(result.a).resize(result.b + 1);
    }
    if (buckets.at(result.a).at(result.b).size() <= result.c) {
        buckets.at(result.a).at(result.b).resize(result.c + 1);
    }
    buckets.at(result.a).at(result.b).at(result.c).push_back(p);
    
    if (buckets.at(result.a).at(result.b).at(result.c).size() == 1) {
        groupedBuckets.push_back(&buckets.at(result.a).at(result.b).at(result.c));
        addedBuckets.push_back({&buckets.at(result.a).at(result.b).at(result.c),
            result});
        addCount++;
    }
    
#ifndef DEBUG
    // Check if particle belongs in bucket to the left
    if (resultMinus.a != result.a) {
        if (buckets.size() <= resultMinus.a) {
            buckets.resize(resultMinus.a + 1);
        }
        if (buckets.at(resultMinus.a).size() <= result.b) {
            buckets.at(resultMinus.a).resize(result.b + 1);
        }
        if (buckets.at(resultMinus.a).at(result.b).size() <= result.c) {
            buckets.at(resultMinus.a).at(result.b).resize(result.c + 1);
        }
        buckets.at(resultMinus.a).at(result.b).at(result.c).push_back(p);
        
        if (buckets.at(resultMinus.a).at(result.b).at(result.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(resultMinus.a).at(result.b).at(result.c));
            addedBuckets.push_back({&buckets.at(resultMinus.a).at(result.b).at(result.c),
                {resultMinus.a, result.b, result.c}});
            addCount++;
        }
    }
    // Check if particle belongs in bucket below
    if (resultMinus.b != result.b) {
        if (buckets.at(result.a).size() <= resultMinus.b) {
            buckets.at(result.a).resize(resultMinus.b + 1);
        }
        if (buckets.at(result.a).at(resultMinus.b).size() <= result.c) {
            buckets.at(result.a).at(resultMinus.b).resize(result.c + 1);
        }
        buckets.at(result.a).at(resultMinus.b).at(result.c).push_back(p);
        
        if (buckets.at(result.a).at(resultMinus.b).at(result.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(result.a).at(resultMinus.b).at(result.c));
            addedBuckets.push_back({&buckets.at(result.a).at(resultMinus.b).at(result.c),
                {result.a, resultMinus.b, result.c}});
            addCount++;
        }
    }
    // Check if particle belongs in bucket behind
    if (resultMinus.c != result.c) {
        if (buckets.at(result.a).at(result.b).size() <= resultMinus.c) {
            buckets.at(result.a).at(result.b).resize(result.c + 1);
        }
        buckets.at(result.a).at(result.b).at(resultMinus.c).push_back(p);
        
        if (buckets.at(result.a).at(result.b).at(resultMinus.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(result.a).at(result.b).at(resultMinus.c));
            addedBuckets.push_back({&buckets.at(result.a).at(result.b).at(resultMinus.c),
                {result.a, result.b, resultMinus.c}});
            addCount++;
        }
    }
    
    // Check if particle belongs in bucket to the right
    if (resultPlus.a != result.a) {
        if (buckets.size() <= resultPlus.a) {
            buckets.resize(resultPlus.a + 1);
        }
        if (buckets.at(resultPlus.a).size() <= result.b) {
            buckets.at(resultPlus.a).resize(result.b + 1);
        }
        if (buckets.at(resultPlus.a).at(result.b).size() <= result.c) {
            buckets.at(resultPlus.a).at(result.b).resize(result.c + 1);
        }
        buckets.at(resultPlus.a).at(result.b).at(result.c).push_back(p);
        
        if (buckets.at(resultPlus.a).at(result.b).at(result.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(resultPlus.a).at(result.b).at(result.c));
            addedBuckets.push_back({&buckets.at(resultPlus.a).at(result.b).at(result.c),
                {resultPlus.a, result.b, result.c}});
            addCount++;
        }
    }
    // Check if particle belongs in bucket above
    if (resultPlus.b != result.b) {
        if (buckets.at(result.a).size() <= resultPlus.b) {
            buckets.at(result.a).resize(resultPlus.b + 1);
        }
        if (buckets.at(result.a).at(resultPlus.b).size() <= result.c) {
            buckets.at(result.a).at(resultPlus.b).resize(result.c + 1);
        }
        buckets.at(result.a).at(resultPlus.b).at(result.c).push_back(p);
        
        if (buckets.at(result.a).at(resultPlus.b).at(result.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(result.a).at(resultPlus.b).at(result.c));
            addedBuckets.push_back({&buckets.at(result.a).at(resultPlus.b).at(result.c),
                {result.a, resultPlus.b, result.c}});
            addCount++;
        }
    }
    // Check if particle belongs in bucket in front
    if (resultPlus.c != result.c) {
        if (buckets.at(result.a).at(result.b).size() <= resultPlus.c) {
            buckets.at(result.a).at(result.b).resize(resultPlus.c + 1);
        }
        buckets.at(result.a).at(result.b).at(resultPlus.c).push_back(p);
        
        if (buckets.at(result.a).at(result.b).at(resultPlus.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(result.a).at(result.b).at(resultPlus.c));
            addedBuckets.push_back({&buckets.at(result.a).at(result.b).at(resultPlus.c),
                {result.a, result.b, resultPlus.c}});
            addCount++;
        }
    }
#endif

    // this is a bit hacky and probably super slow, but if a pointer changes,
    // we need to update the corresponding pointer in groupedBuckets
    for (int i = 0; i < groupedBuckets.size(); i++) {
        auto b = groupedBuckets.at(i);

        if (b->size() == 0) {
            Triplet t = addedBuckets.at(i).t;
            Bucket_t *p = &buckets.at(t.a).at(t.b).at(t.c);
            
            groupedBuckets.at(i) = p;
            addedBuckets.at(i).p = p;
        }
    }
}

void Hash::clear() {
    for (int i = 0; i < buckets.size(); i++) {
        if (buckets.at(i).size() > 0) {
            int size = (int) buckets.at(i).size();
            for (int j = 0; j < size; j++) {
                buckets.at(i).at(j).clear();
            }
        }
    }
    
    groupedBuckets.clear();
}

void Hash::threadStep(int threadId) {
    int size =  (int) groupedBuckets.size();
    int span = (size + NUM_THREADS - 1) / NUM_THREADS;
    int start = threadId * span;
    int end = start + span;

    for (int k = start; k < end && k < size; k++) {
        Bucket_t *bucket = groupedBuckets.at(k);
        
        // We want to check that this bucket didn't somehow get messed up. It's not worth
        // it to try and recover, so we're just going to bail on this bucket this frame
        if (bucket->size() > NUM_PARTICLES) {
            return;
        }
        
        for (int i = 0; i < bucket->size(); i++) {
            auto a = bucket->at(i);
            
            for (int j = i + 1; j < bucket->size(); j++) {
                auto b = bucket->at(j);
                
                float dist = a->distance(b);
                // check if particles are close
                if (dist < RADIUS) {
                    Vector3f dir = (b->x - a->x).normalized(); // direction from a to b
                    // check if particles are approaching each other
                    if ((b->v - a->v).dot(dir) < 0) {
                        a->v -= dir * VISCOSITY_GAIN;
                        b->v += dir * VISCOSITY_GAIN;
                    }
                    // if particles are separating, compute cohesion force
                    else {
                        // force attracting a to b
                        Vector3f cohesion, repulsion;
                        
                        cohesion = (RADIUS - dist) / RADIUS * dir * COHESION;
                        repulsion = (1 - (RADIUS - dist) / RADIUS) * -dir;
                        a->addForce(cohesion);
                        a->addForce(repulsion);
                        b->addForce(-cohesion);
                        b->addForce(-repulsion);
                    }
                }
            }
        }
    }
}

void Hash::colorBuckets() {
    Vector3f colors[] = {
        Vector3f(1, 1, 1),
        Vector3f(0, 0, 0)
    };
 
    for (int i = 0; i < groupedBuckets.size(); i++) {
        Triplet t = hash(groupedBuckets.at(i)->at(0));
 
        int colorIndex = (t.a % 2);
        if (t.b % 2) {
            colorIndex = !colorIndex;
        }
        if (t.c % 2) {
            colorIndex = !colorIndex;
        }
        
        for (int j = 0; j < groupedBuckets.at(i)->size(); j++) {
            groupedBuckets.at(i)->at(j)->color = colors[colorIndex];
        }
    }
}

void Hash::step() {
#ifdef DEBUG
    colorBuckets();
#endif
    
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId] = thread( [this, threadId]
                                            { threadStep(threadId); }
                                         );
    }
    
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId].join();
    }
    
    addCount = 0;
    addedBuckets.clear();
}

