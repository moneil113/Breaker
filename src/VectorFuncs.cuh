#ifndef _VECTOR_FUNCS_CUH_
#define _VECTOR_FUNCS_CUH_
#include <vector_types.h>
// #include <vector_functions.h>

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

#endif
