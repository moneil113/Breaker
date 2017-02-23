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
//        cout << result.c << " vs " << buckets.at(result.a).at(result.b).size() << endl;
//        cout << "(" << p->x.x() << ", " << p->x.y() << ", " << p->x.z() << ")" << endl;
        buckets.at(result.a).at(result.b).resize(result.c + 1);
    }
    
    buckets.at(result.a).at(result.b).at(result.c).push_back(p);
    
    if (buckets.at(result.a).at(result.b).at(result.c).size() == 1) {
        groupedBuckets.push_back(&buckets.at(result.a).at(result.b).at(result.c));
        addCount++;
    }
    
#ifndef DEBUG
    // Check if particle belongs in bucket to the left or below
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
            addCount++;
        }
    }
    if (resultMinus.b != result.b) {
        if (buckets.size() <= result.a) {
            buckets.resize(result.a + 1);
        }
        if (buckets.at(result.a).size() <= resultMinus.b) {
            buckets.at(result.a).resize(resultMinus.b + 1);
        }
        if (buckets.at(result.a).at(resultMinus.b).size() <= result.c) {
            buckets.at(result.a).at(resultMinus.b).resize(result.c + 1);
        }
        buckets.at(result.a).at(resultMinus.b).at(result.c).push_back(p);
        
        if (buckets.at(result.a).at(resultMinus.b).at(result.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(result.a).at(resultMinus.b).at(result.c));
            addCount++;
        }
    }
    
    // Check if particle belongs in bucket to the right or above
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
            addCount++;
        }
    }
    if (resultPlus.b != result.b) {
        if (buckets.size() <= result.a) {
            buckets.resize(result.a + 1);
        }
        if (buckets.at(result.a).size() <= resultPlus.b) {
            buckets.at(result.a).resize(resultPlus.b + 1);
        }
        if (buckets.at(result.a).at(resultPlus.b).size() <= result.c) {
            buckets.at(result.a).at(resultPlus.b).resize(result.c + 1);
        }
        buckets.at(result.a).at(resultPlus.b).at(result.c).push_back(p);
        
        if (buckets.at(result.a).at(resultPlus.b).at(result.c).size() == 1) {
            groupedBuckets.push_back(&buckets.at(result.a).at(resultPlus.b).at(result.c));
            addCount++;
        }
    }
#endif
}

void Hash::clear() {
    for (int i = 0; i < buckets.size(); i++) {
        if (buckets.at(i).size() > 0) {
            int size = buckets.at(i).size();
            for (int j = 0; j < size; j++) {
                buckets.at(i).at(j).clear();
            }
        }
    }
    
    groupedBuckets.clear();
}

void Hash::threadStep(int threadId) {
    int size = groupedBuckets.size();
    int span = (size + NUM_THREADS - 1) / NUM_THREADS;
    int start = threadId * span;
    int end = start + span;

    for (int k = start; k < end && k < size; k++) {
        Bucket_t *bucket = groupedBuckets.at(k);
        for (int i = 0; i < bucket->size() - 1; i++) {
            shared_ptr<Particle> a = bucket->at(i);
            for (int j = i + 1; j < bucket->size(); j++) {
                shared_ptr<Particle> b = bucket->at(j);
                // check if particles are close
                if (a->distance2(b) < EPSILON) {
                    Vector3f dir = (a->x - b->x).normalized(); // direction from b to a
                    if ((a->v - b->v).dot(dir) < 0) {
                        stringstream d2;
                        // particles are approaching each other
                        a->v += dir * VISCOSITY_GAIN;
                        b->v -= dir * VISCOSITY_GAIN;
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
        
        for (int j = 0; j < groupedBuckets.at(i)->size(); j++) {
            groupedBuckets.at(i)->at(j)->color = colors[colorIndex];
        }
    }
}

void Hash::step() {
#ifdef DEBUG
    colorBuckets();
#endif
    
    
    // TODO this implementation doesn't synchronize threads when it deals with
    // particles in more than one bucket
    // this situation is relatively rare (I think), so we aren't going to worry
    // about it right now
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId] = thread( [this, threadId]
                                            { threadStep(threadId); }
                                         );
    }
    
    for (int threadId = 0; threadId < NUM_THREADS; threadId++) {
        threadHandles[threadId].join();
    }
    
    addCount = 0;
}

