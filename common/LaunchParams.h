#ifndef LAUNCHPARAMS_H
#define LAUNCHPARAMS_H

#include "optix7.h"
#include "CUDABuffer.h"
#include "vec.h"
#include "camera.h"
#include <stddef.h>

#define LP64
enum RENDER_TYPE{MIP};

enum OBJECT_TYPE {VOLUME_RENDERING=0,MESH_RENDERING};
//Cette structure permet de facilement faire la liaison binaire entre un float et un unsigned int.
//Elle est utilisé pour passer les données concernant les temps d'intersection sur le gpu
typedef struct time {
  union {
      float ftmin;
      unsigned int uitmin;
  }tmin;

  union {
      float ftmax;
      unsigned int uitmax;
  }tmax;

}intersection_time;

//Cette structure représente différentes informations de la scène.
/*
    Elle est composé de 3 parties :
        -Traversable : l'objet nécessaire à optix pour calculer les intersections
        -Camera : une définition de la caméra gérée par la caméra OpenGL.
        -Frame : différentes extra-informations pour le rendu
*/
struct LaunchParams
{
    struct {
        uint32_t* colorBuffer; //resultat du rendu
        vec3f lightPosition;
        vec2i     size; // taille de l'image
        float sampler = 500.f;
        float ki = -1;
        float ks = -1; 
        float minIntensity = 0.0f;
        float maxIntensity = 1.0f;
        unsigned char renderType = MIP; 
    } frame;

    struct {
        vec3f position;
        vec3f direction;
        vec3f horizontal;
        vec3f vertical;
        vec3f up;
    } camera;
    OptixTraversableHandle traversable;

};
/*
    La structure VolumetricCube permet de faire la liaison entre l'Objet Volume utilisé sur le host et
    et ses informations sur le device.
*/
struct VolumetricCube {
     vec3f size;
     vec3f center;
     vec3i sizePixel;
     cudaTextureObject_t tex = 0;
     vec3f color;
 };
/*
    La structure TriangleMeshSBT permet de faire la liaison entre l'Objet TriangleMesh utilisé sur le host et
    et ses informations sur le device.
*/
struct TriangleMeshSBT {
    vec3f kd;
    vec3f *vertex;
    vec3i *indices;
    vec2f *texCoord;
    cudaTextureObject_t tex = 0;
    unsigned char hasTexture = 0;
 };  
    

/*
    La structure sbtData permet de regrouper l'ensemble des informations pour afficher
    l'ensemble des éléments affichables dans notre scène.
    Il est constitué de l'ensemble des structures précédentes
*/
struct sbtData {
        VolumetricCube volumeData;
        TriangleMeshSBT meshData;
};


#endif
