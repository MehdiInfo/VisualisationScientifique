#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <vector>
#include "trianglemesh.h"
#include "volume.h"

class Scene {
    public: 
    Scene();

    /*Ajout d'objets dans la scène*/
    void addMesh(TriangleMesh* mesh);
    void addVolume(Volume* v);
    /*Suppression d'objet dans la scène*/
    void rmMesh(const int id);
    void rmVolume(const int id);
    void clear();

    size_t getNumVolume() const;
    size_t getNumMesh() const;

    void getVolumeSbt(const int i, sbtData *sbt);
    void getMeshSbt(const int i, sbtData *sbt);

    void getAabb(const int i, float* aabb);

    TriangleMesh* getMesh(const int id);
    private : 
    std::vector<TriangleMesh*> m_meshs;
    std::vector<Volume*> m_volumes;
};

#endif
