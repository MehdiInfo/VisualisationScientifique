#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cmath>
#include "vec.h"
#include <cstdlib>
#include <iostream>


struct Matrix3x3{
    float m_data[9];

    Matrix3x3(){
        for(int i = 0; i < 3 ; ++i)
            for(int j = 0; j < 3 ; ++j)
                m_data[i * 3 + j ] = i==j ? 1 : 0;
    }
    Matrix3x3(const float data[9]){
        for(int i = 0; i < 9 ; ++i)
            m_data[i] = data[i];
    }
    Matrix3x3 rotationX(const float angle);
    Matrix3x3 rotationY(const float angle);
    Matrix3x3 rotationZ(const float angle);

};
Matrix3x3 operator* (const Matrix3x3 &m1, const Matrix3x3 &m2);
vec3f operator* (const Matrix3x3 &m1, const vec3f v);
    

#endif