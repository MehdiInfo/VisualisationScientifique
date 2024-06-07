#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include <vector>
#include <iostream>
#include "vec.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"

/**
    \class TriangleMesh
    \brief Représente un maillage triangulaire.
    Un maillage triangulaire est basiquement définis par une liste d'indices représentant les faces et un ensemble de sommet.
    Nous rajoutons une taille et un centre pour pouvoir dimensionner et centrer l'ensemble des sommets.
    De plus, les coordonnées UV et les textures sont sauvegardées.
*/
class TriangleMesh
{
public:
    TriangleMesh();
    ~TriangleMesh();

    /**
        \brief Creer un maillage d'un plan ( soit 2 triangles)
    */
    void addPlane(vec3f &center, vec3f &size, vec3f &color);
    void addUnitCube();

    /**
        \brief Retourne le nombre de sommet
    */
    size_t getNumVertices() const;

    /**
        \brief Retourne le nombre d'indice
    */
    size_t getNumIndex() const;

    /**
        \brief Retourne un pointeur pour le gpu qui pointe vers le tableau des sommets
    */
    CUdeviceptr getVertexDevicePointer();

    /**
        \brief Retourne un pointeur pour le gpu qui pointe vers le tableau des indices
    */
    CUdeviceptr getIndexDevicePointer();
     /**
        \brief Retourne un pointeur pour le gpu qui pointe vers le tableau des textures
    */
    CUdeviceptr getTextureCoordinateBuffer();

    void setTexture(cudaTextureObject_t texture);
    /**
        \brief Retourne la structure GPU qui definis un maillage triangulaire
    */
    void getSBT(sbtData *sbt);

    /**
        \brief Ajoutes un sommets
    */
    void addVertices(const std::vector<vec3f> vertices);
    /**
        \brief Ajoutes un indice de sommet
    */
    void addIndices(const std::vector<vec3i> indices);
    void addTextureCoordinate(const std::vector<vec2f> texCoords);
    /**
        \brief Realise la translation d'un maillage triangulaire
    */
    virtual void translate(const vec3f &t);

    /**
        \brief Set la taille de l'objet par la valeur newDim
    */
    virtual void resize(const vec3f &newDim);

    /**
        \brief Retourne le centre de l'objet
    */
    vec3f getCenter();

    /**
        \brief Retourne la taille de l'objet
    */
    vec3f getSize();

    void setColor(vec3f &c);
private :
    vec3f color; //color de base
    vec3f size; //taille de la bbox
    vec3f center; // centre de la bbox

    std::vector<vec3i> index; //liste des indices ---> represente les faces
    std::vector<vec3f> vertex; //listes des sommets
    std::vector<vec2f> texCoord; //Coordonnees textures

    CUDABuffer vertexBuffer; //listes des sommets
    CUDABuffer indexBuffer; 
    CUDABuffer texCoordBuffer; // listes des coordonnees textures par GPU
    cudaTextureObject_t tex = 0; //objet representant une texture pour Optix par GPU
    
};

#endif // TRIANGLEMESH_H
