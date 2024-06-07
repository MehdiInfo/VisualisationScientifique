#ifndef VEC_H
#define VEC_H

#include <cuda.h>
#ifndef DEVICE
#define __both__ __host__
#endif

#ifdef DEVICE
#define __both__ __device__
#endif

#include <cmath>
#include <cuda.h>
#include <optix.h>
#include <cuda_runtime.h>

//#include <cuda.h>
/*
    Ce fichier comprends un ensemble de fonction possible concernant les vecteurs.
    Il nous permet d'avoir des types représentant des vecteurs 1D,2D,3D,4D avec l'ensemble des types que nous souhaitons.
    De plus, ces structures sont utilisables sur le host ainsi que sur les devices.
    Diverse opérateurs sont implémentées : +,-,*,/
    et également diverses fonctions : norme, dot, normalize.
*/

template<typename T, int N>  struct vec_t;

template<typename T>
struct vec_t<T,1>{
    __both__ vec_t<T,1>(){x = 0;}
    __both__ vec_t<T,1>(T val){x = val;}
    T x;
};


template<typename T>
struct vec_t<T,2>{
    __both__ vec_t<T,2>():x(0),y(0){}
    __both__ vec_t<T,2>(vec_t<T,2> const &v):x(v.x),y(v.y){}
    __both__ vec_t<T,2>(T val){x = val; y = val;}
    __both__ vec_t<T,2>(T dx, T dy){x = dx; y = dy;}
    T x,y;
};


template<typename T>
struct vec_t<T,3>{
    __both__ vec_t<T,3>(){x = 0;}
    __both__ vec_t<T,3>(T val){x = val; y = val; z = val;}
    __both__ vec_t<T,3>(T dx, T dy, T dz){x = dx; y = dy; z = dz;}
    __both__ vec_t<T,3>(float3 f3):x(f3.x),y(f3.y),z(f3.z){}


    T x,y,z;
};
template<typename T>
struct vec_t<T,4>{
    __both__ vec_t<T,4>(){x = 0;}
    __both__ vec_t<T,4>(T val){x = val; y = val; z = val; w = val;}
    __both__ vec_t<T,4>(T dx, T dy, T dz, T dw){x = dx; y = dy; z = dz; w = dw;}

    __both__ T& operator()(const int i)
    {
        T res;
        switch(i){
            case 0 :
                res = x;
            break;
            case 1 :
                res = y;
            break;
            case 2 :
                res = z;
            break;
            case 3 :
                res = w;
            break;
        }
        return res;
    }
    T x,y,z,w;
};

typedef vec_t<float,2> vec2f;
typedef vec_t<float,3> vec3f;
typedef vec_t<float,4> vec4f;

typedef vec_t<int,2> vec2i;
typedef vec_t<int,3> vec3i;
typedef vec_t<int,4> vec4i;



/*Operator+ */
template<typename T> __both__
 vec_t<T,1> operator+ (const vec_t<T,1> &v1, const vec_t<T,1> &v2){
    return vec_t<T,1>(v1.x + v2.x);
}
template<typename T> __both__
vec_t<T,2> operator+ (const vec_t<T,2> &v1, const vec_t<T,2>& v2){
    return vec_t<T,2>(v1.x + v2.x, v1.y + v2.y);
}
template<typename T> __both__
vec_t<T,3> operator+ (const vec_t<T,3> &v1, const vec_t<T,3> &v2){
    return vec_t<T,3>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

template<typename T> __both__
vec_t<T,1> operator+ (const vec_t<T,1> &v1, const T &v){
    return vec_t<T,1>(v1.x + v);
}
template<typename T> __both__
vec_t<T,2> operator+ (const vec_t<T,2> &v1, const T &v){
    return vec_t<T,2>(v1.x + v, v1.y + v);
}
template<typename T> __both__
vec_t<T,3> operator+ (const vec_t<T,3> &v1, const float &v){
    return vec_t<T,3>(v1.x + v, v1.y + v, v1.z + v);
}
//Operator /
template<typename T> __both__
vec_t<T,1> operator/ (const vec_t<T,1> &v1, const vec_t<T,1> &v2){
    return vec_t<T,1>(v1.x / v2.x);
}
template<typename T> __both__
vec_t<T,2> operator/ (const vec_t<T,2> &v1, const vec_t<T,2>& v2){
    return vec_t<T,2>(v1.x / v2.x, v1.y / v2.y);
}
template<typename T> __both__
vec_t<T,3> operator/ (const vec_t<T,3> &v1, const vec_t<T,3> &v2){
    return vec_t<T,3>(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

template<typename T> __both__
vec_t<T,1> operator/ (const vec_t<T,1> &v1, const T &v){
    return vec_t<T,1>(v1.x / v);
}
template<typename T> __both__
vec_t<T,2> operator/ (const vec_t<T,2> &v1, const T &v){
    return vec_t<T,2>(v1.x / v, v1.y / v);
}
template<typename T> __both__
vec_t<T,3> operator/ (const vec_t<T,3> &v1, const float &v){
    return vec_t<T,3>(v1.x / v, v1.y / v, v1.z / v);
}

//Operator *
template<typename T> __both__
vec_t<T,1> operator* (const vec_t<T,1> &v1, const vec_t<T,1> &v2){
    return vec_t<T,1>(v1.x * v2.x);
}
template<typename T> __both__
vec_t<T,2> operator* (const vec_t<T,2> &v1, const vec_t<T,2>& v2){
    return vec_t<T,2>(v1.x * v2.x, v1.y * v2.y);
}
template<typename T> __both__
vec_t<T,3> operator* (const vec_t<T,3> &v1, const vec_t<T,3> &v2){
    return vec_t<T,3>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

template<typename T> __both__
vec_t<T,1> operator* (const vec_t<T,1> &v1, const T &v){
    return vec_t<T,1>(v1.x * v);
}
template<typename T> __both__
vec_t<T,2> operator* (const vec_t<T,2> &v1, const T &v){
    return vec_t<T,2>(v1.x * v, v1.y * v);
}
template<typename T> __both__
vec_t<T,3> operator* (const vec_t<T,3> &v1, const float &v){
    return vec_t<T,3>(v1.x * v, v1.y * v, v1.z * v);
}
template<typename T> __both__
vec_t<T,1> operator* (const T &v,const vec_t<T,1> &v1 ){
    return vec_t<T,1>(v1.x * v);
}
template<typename T> __both__
vec_t<T,2> operator* (const T &v,const vec_t<T,2> &v1){
    return vec_t<T,2>(v1.x * v, v1.y * v);
}
template<typename T> __both__
vec_t<T,3> operator* (const T &v, const vec_t<T,3> &v1){
    return vec_t<T,3>(v1.x * v, v1.y * v, v1.z * v);
}




//Operator -

template<typename T> __both__
vec_t<T,1> operator- (const vec_t<T,1> &v1, const vec_t<T,1> &v2){
    return vec_t<T,1>(v1.x - v2.x);
}
template<typename T> __both__
vec_t<T,2> operator- (const vec_t<T,2> &v1, const vec_t<T,2>& v2){
    return vec_t<T,2>(v1.x - v2.x, v1.y - v2.y);
}
template<typename T> __both__
vec_t<T,3> operator- (const vec_t<T,3> &v1, const vec_t<T,3> &v2){
    return vec_t<T,3>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

template<typename T> __both__
vec_t<T,1> operator- (const vec_t<T,1> &v1, const T &v){
    return vec_t<T,1>(v1.x - v);
}
template<typename T> __both__
vec_t<T,2> operator- (const vec_t<T,2> &v1, const T &v){
    return vec_t<T,2>(v1.x - v, v1.y - v);
}
template<typename T> __both__
vec_t<T,3> operator- (const vec_t<T,3> &v1, const float &v){
    return vec_t<T,3>(v1.x - v, v1.y - v, v1.z - v);
}
template<typename T> __both__
vec_t<T,1> operator- (const T &v,const vec_t<T,1> &v1 ){
    return vec_t<T,1>(v1.x - v);
}
template<typename T> __both__
vec_t<T,2> operator- (const T &v,const vec_t<T,2> &v1){
    return vec_t<T,2>(v1.x - v, v1.y - v);
}
template<typename T> __both__
vec_t<T,3> operator- (const T &v, const vec_t<T,3> &v1){
    return vec_t<T,3>(v1.x - v, v1.y - v, v1.z - v);
}
//Norme
template<typename T> __both__
float norme(vec_t<T,2> v){
   return sqrt(v.x * v.x + v.y * v.y);
}
template<typename T> __both__
float norme(vec_t<T,3> v){
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

//Normalize
template<typename T> __both__
vec_t<T,2> normalize(vec_t<T,2> v){
    return v / norme(v);
}
template<typename T> __both__
vec_t<T,3> normalize(vec_t<T,3> v){
    return v / norme(v);
}
/*template<typename T> __both__
vec_t<T,3> normalize(float3 v){
    vec3f v3 = v;
    return v3 / norme(v3);
}*/

//Cross

/*! vector cross product */
template<typename T> __both__
inline vec_t<T,3> cross(const vec_t<T,3> &a, const vec_t<T,3> &b)
{
    return vec_t<T,3>(a.y*b.z-b.y*a.z,
                      a.z*b.x-b.z*a.x,
                      a.x*b.y-b.x*a.y);
}
template<typename T> __both__
T dot(const vec_t<T,3> &a, const vec_t<T,3> &b)
{
 return a.x*b.x + a.y*b.y + a.z*b.z;
}
template<typename T> __both__
T dot(const vec_t<T,2> &a, const vec_t<T,2> &b)
{
 return a.x*b.x + a.y*b.y;
}

template<typename T> __both__
inline vec_t<T,3> minVec(const vec_t<T,3> &a, const vec_t<T,3> &b)
{
    vec3f r;
    r.x = a.x < b.x ? a.x : b.x;
    r.y = a.y < b.y ? a.y : b.y;
    r.z = a.z < b.z ? a.z : b.z;
    return r;
}
template<typename T> __both__
inline vec_t<T,3> maxVec(const vec_t<T,3> &a, const vec_t<T,3> &b)
{
    vec3f r;
    r.x = a.x > b.x ? a.x : b.x;
    r.y = a.y > b.y ? a.y : b.y;
    r.z = a.z > b.z ? a.z : b.z;
    return r;
}

template<typename T> __both__
inline T getMin(const vec_t<T,3> &a)
{
    float r = a.x;
    r = r < a.y ? r : a.y;
    r = r < a.z ? r : a.z;
    return r;
}
template<typename T> __both__
inline T getMax(const vec_t<T,3> &a)
{
    float r = a.x;
    r = r > a.y ? r : a.y;
    r = r > a.z ? r : a.z;
    return r;
}
template<typename T> __both__
inline T getMax(const vec_t<T,4> &a)
{
    float r = a.x;
    r = r > a.y ? r : a.y;
    r = r > a.z ? r : a.z;
    r = r > a.w ? r : a.w;
    return r;
}

template<typename T> __both__
inline vec_t<T,3> dotVectoriel(const vec_t<T,3> &a,const vec_t<T,3> &b)
{
    vec3f res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}




#endif // VEC_H

