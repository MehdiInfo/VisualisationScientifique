#include "mmatrix.h"

Matrix3x3 Matrix3x3::rotationX(const float angle){
    Matrix3x3 r;
    r.m_data[4] = cos(angle);
    r.m_data[5] = -sin(angle);
    r.m_data[7] = sin(angle);
    r.m_data[8] = cos(angle);
    return r;
}
Matrix3x3 Matrix3x3::rotationY(const float angle){
    Matrix3x3 r;
    r.m_data[0] = cos(angle);
    r.m_data[2] = sin(angle);
    r.m_data[6] = -sin(angle);
    r.m_data[8] = cos(angle);
    return r;
}
Matrix3x3 Matrix3x3::rotationZ(const float angle){
    Matrix3x3 r;
    r.m_data[0] = cos(angle);
    r.m_data[1] = -sin(angle);
    r.m_data[3] = sin(angle);
    r.m_data[4] = cos(angle);
    return r;
}

Matrix3x3 operator* (const Matrix3x3 &m1, const Matrix3x3 &m2){
    Matrix3x3 r;
    for(int i = 0; i < 3 ; ++i){
        for(int j  = 0 ; j < 3 ; j++){
            r.m_data[i*3 + j] = 0;
            for(int k = 0; k < 3 ; ++k){
                r.m_data[i*3 + j] += m1.m_data[i*3+j] * m2.m_data[i*3 + k];
            }
        }
    }
    return r;
}

vec3f operator* (const Matrix3x3 &m1, const vec3f v){
    vec3f r = vec3f(0.f);
    r.x += m1.m_data[0] * v.x + m1.m_data[1] * v.y + m1.m_data[2] * v.z;
    r.y += m1.m_data[3] * v.x + m1.m_data[4] * v.y + m1.m_data[5] * v.z;
    r.z += m1.m_data[6] * v.x + m1.m_data[7] * v.y + m1.m_data[8] * v.z;
    return r;
}