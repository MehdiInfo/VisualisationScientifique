#include "Scene.hpp"

Scene::Scene(){}
/*Ajout d'objets dans la scène*/
void Scene::addMesh(TriangleMesh* mesh){
    m_meshs.push_back(mesh);
}

void Scene::addVolume(Volume* v){
    m_volumes.push_back(v);
}
/*Suppression d'objet dans la scène*/
    
void Scene::rmVolume(const int id){
    m_volumes.erase(m_volumes.begin() + id);
}

void Scene::rmMesh(const int id){
    m_meshs.erase(m_meshs.begin() + id);
}

void Scene::clear(){
    m_meshs.clear();
    m_volumes.clear();
}

size_t Scene::getNumVolume() const{
    return m_volumes.size();
}

size_t Scene::getNumMesh() const{
    return m_meshs.size();
}

void Scene::getVolumeSbt(const int i, sbtData *sbt){
    if( i < m_volumes.size() )
        m_volumes[i]->getSbt(sbt);
}

void Scene::getMeshSbt(const int i, sbtData *sbt){
    if( i < m_meshs.size())
        m_meshs[i]->getSBT(sbt);
}


TriangleMesh* Scene::getMesh(const int id){
    if (id < m_meshs.size())
        return m_meshs[id];
    return nullptr;
}

void Scene::getAabb(const int i, float* aabb){
    if( i < m_volumes.size() )
        m_volumes[i]->getAabb(aabb);
}


