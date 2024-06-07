#ifndef CAMERA_H
#define CAMERA_H

#include "vec.h"

typedef struct Camera {
    Camera(vec3f &newPos, vec3f &newAt, vec3f &newUp) : pos(newPos),at(newAt), up(newUp){}
    Camera(){
	pos = vec3f(0.f,0.f,5.f);
	at = vec3f(0.f,0.f,0.f);
	up = vec3f(0.f,1.f,0.f);
    }

    vec3f getAt() const       {return at ;}
    vec3f getUp() const       {return up ;}
    vec3f getPosition() const {return pos;}

    vec3f at, up,  pos;    
}Camera;

#endif
