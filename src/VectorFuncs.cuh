#ifndef _VECTOR_FUNCS_CUH_
#define _VECTOR_FUNCS_CUH_
#include <vector_types.h>

// float3
inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __device__ void operator-=(float3 &a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 normalize(float3 a) {
    return a / sqrtf(dot(a, a));
}

// int3
inline __device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

#endif
