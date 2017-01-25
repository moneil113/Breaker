//
//  Hash.cpp
//  Breaker
//
//  Created by Matthew O'Neil on 9/11/16.
//
//

#include "Hash.h"
#include "Particle.h"

#include <iostream>

using namespace std;
using namespace Eigen;

typedef struct Triplet {
    int a;
    int b;
    int c;
} Triplet;

Hash::Hash(float bucketSize) {
    this->bucketSize = bucketSize;
    buckets = vector<vector<vector<vector<shared_ptr<Particle>>>>>();
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
    Triplet resultMinus = hashMinus(p);
    Triplet resultPlus = hashPlus(p);

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
    }
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
}

void Hash::stepBucket(Triplet t) {
    if (buckets.at(t.a).at(t.b).size() == 0) {
        return;
    }
    
    int size = buckets.at(t.a).at(t.b).at(t.c).size();
    for (int i = 0; i < size; i++) {
        if (size == 0) {
            // this shouldn't happen but if it does then we just want to bail
            assert(0);
        }
        auto a = buckets.at(t.a).at(t.b).at(t.c).at(i);
        for (int j = i + 1; j < size; j++) {
            auto b = buckets.at(t.a).at(t.b).at(t.c).at(j);
            // check if particles are close
            if (a->distance2(b) < EPSILON) {
                Vector3f dir = (a->x - b->x).normalized(); // direction from b to a
                if ((a->v - b->v).dot(dir) < 0) {
                    // particles are approaching each other
                    a->v += dir * VISCOSITY_GAIN;
                    b->v -= dir * VISCOSITY_GAIN;
                }
            }
        }
    }
}

void Hash::step() {
    for (int i = 0; i < buckets.size(); i++) {
        if (buckets.at(i).size() > 0) {
            int size = buckets.at(i).size();
            for (int j = 0; j < size; j++) {
                stepBucket({i, j});
            }
        }
    }
}
