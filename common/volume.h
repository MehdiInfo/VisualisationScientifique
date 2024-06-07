#ifndef VOLUME_H
#define VOLUME_H

#include "vec.h"
#include "CUDABuffer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_texture_types.h>
#include <cuda_runtime_api.h>
#include <texture_fetch_functions.h>
#include <optix_types.h>
#include "LaunchParams.h"
#include <iostream>



/**
    \class Volume
    \brief Represente un Volume
    Un volume est définis par une position dans l'espace, une dimension dans l'espace, une dimension pour les données
    et un ensemble de donnée qui traduis les valeurs des voxels.
    Pour stocker les données, nous utilisons des textures ce qui nous permettent une interpolation tri-linéaire à la récupération des valeurs
    et de meilleurs performances.
*/
class Volume
{
public:
    Volume();
    Volume(std::string path);


   void loadVolume(const std::string &path);

    /**
       \Brief Créer la texture à partir des données chargées.
    */
    void createTexture();

    virtual void resize(const vec3f &newDim);

    /**
       \brief Retourne la dimension dans l'espace
    */
    vec3f getWorldSize() const;

    /**
       \brief Retourne la position dans l'espace
    */
    vec3f getWorldCenter() const;

    /**
       \brief Retourne la dimension en voxel
    */
    vec3i getPixelSize() const;


    /**
       \brief Retourne le CudaBuffer qui pointe vers les données sur le gpu.
    */
    CUDABuffer getCudaBuffer() const;

    /**
       \brief Retourne le pointeur vers les données sur le gpu
    */
    CUdeviceptr getDevicePointer() const;


    void* getData() const { return pData;}
    int getDataType() const {return type;}
    /**
       \brief Retourne le pointeur de la texture 3D représentant les données
    */
    cudaArray_t getTextureDevicePointer() const;

    /**
       \brief Retourne la texture 3d représentant les données
    */
    cudaTextureObject_t getTextureReference() const;
    virtual void getAabb( float results[6]);
    virtual void getSbt(sbtData *sbt);


    /**
       \brief Permet d'envoyer les données de texture au gpu
    */
    void uploadCudaBuffer();
    /**
        \brief Resize le buffer sur le device
    */
    void resizeCudaBuffer();

    void translate(const vec3f &t);

private :
    enum DATA_TYPE{SHORT,UCHAR};
    void *pData;
    unsigned char type;
    CUDABuffer data;
    vec3f worldSize;
    vec3i pixelSize;
    vec3f position;
    unsigned short maxIntensity = 0;
    unsigned short minIntensity = 255;
    OptixAabb* aabb;

    cudaTextureObject_t cudaTexture = 0;
    cudaArray_t d_array = 0;
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc = {};
};

#endif // VOLUME_H
