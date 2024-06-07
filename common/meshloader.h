#ifndef MESH_H
#define	MESH_H

#include <map>
#include <vector>
#include <iostream>
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags

#include "image.h"
#include "trianglemesh.h"

//Représente un sommet de notre maillage
//Possède l'ensemble des informations nécessaire :
//m_pos : position dans l'espace
//m_tex : coordonnée (u,v) de la texture
//m_normal : la normal en ce point
struct Vertex
{
    vec3f m_pos;
    vec2f m_tex;
    vec3f m_normal;

    Vertex() {}

    Vertex(const vec3f& pos, const vec2f& tex, const vec3f& normal)
    {
        m_pos    = pos;
        m_tex    = tex;
        m_normal = normal;
    }
};

/*
    MeshLoader permet de charger un modèle 3D en utilisant Assimp.
    L'object ce charge de lire le fichier donnée, de récupérer l'ensemble des informations necessaire
    puis de generer un object TriangleMesh utilisable dans notre application.
*/
class MeshLoader
{
public:
    MeshLoader();

    ~MeshLoader();

    //Function a appeler pour charger le modele 3D
    bool LoadMesh(const std::string& Filename);

    //Fonction pour convertir notre maillage charge en un TriangleMesh.
    //Nous construisons un TriangleMesh par MeshEntrie. Pour plus d'info sur les meshEntries, se diriger vers Assimp.
    std::vector<TriangleMesh *> toTriangleMeshs();

    //Fonction qui gère la création des textures pour OptiX en passant par Cuda.
    void createTexture(int index);


    //Retourne le nombre de texture
    size_t getNbTexture(){return m_Textures.size();}

    image_t* getTexture(int i){return m_Textures[i];}
    std::string getTextureName(int i){return m_TexturesName[i];}
private:
    //Fonction qui gère la lecture de la scène
    bool InitFromScene(const aiScene* pScene, const std::string& Filename);

    //Initialise un objet par mesh entries
    void InitMesh(unsigned int Index, const aiMesh* paiMesh);
    //Récupère les informations de textures
    bool InitMaterials(const aiScene* pScene, const std::string& Filename);

    //Reset l'objet
    void Clear();

#define INVALID_MATERIAL 0xFFFFFFFF

    std::vector<std::string> meshEntriName; //noms des differentes parties du maillage
    std::vector<std::vector<Vertex>> m_EntriesVertex; //l'ensemble des sommets du maillages pour chaque entree
    std::vector<std::vector<vec3i>> m_EntriesIndices; //l'ensemble des indices des sommets d'une face pour chaque entree
    std::vector<int> materialIndex; //indice de texture pour chaque entree
    const aiScene* scene; //objet Assimp representant le fichier lu
    std::vector<image_t*> m_Textures; //Ensemble des textures du maillages
    std::vector<std::string> m_TexturesName; //Ensemble des noms des textures du maillage
    cudaArray_t d_array; // pointeur vers la position de la texture.
    std::vector<cudaTextureObject_t> cudaTexture; // l'objet qui represente la texture en cuda/optix
};


#endif	/* MESH_H */

