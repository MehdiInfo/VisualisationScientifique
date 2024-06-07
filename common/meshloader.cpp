#include <assert.h>

#include "meshloader.h"


MeshLoader::MeshLoader()
{
}


MeshLoader::~MeshLoader()
{
    Clear();
}


void MeshLoader::Clear()
{
    std::cout << "Clear mesh" << std::endl;
}


bool MeshLoader::LoadMesh(const std::string& Filename)
{
    // Release the previously loaded mesh (if it exists)
    Clear();

    bool Ret = false;
   Assimp::Importer Importer;

   scene = Importer.ReadFile(Filename.c_str(),  aiProcess_FlipWindingOrder |
                             aiProcess_FlipUVs |
                             aiProcess_PreTransformVertices |
                             aiProcess_CalcTangentSpace |
                             aiProcess_GenSmoothNormals |
                             aiProcess_Triangulate |
                             aiProcess_FixInfacingNormals |
                             aiProcess_FindInvalidData |
                             aiProcess_ValidateDataStructure  |
                             aiProcess_JoinIdenticalVertices |
                              0);
    if (scene) {
        Ret = InitFromScene(scene, Filename);
    }
    else {
        printf("Error parsing '%s': '%s'\n", Filename.c_str(), Importer.GetErrorString());
    }
    return Ret;
}

bool MeshLoader::InitFromScene(const aiScene* pScene, const std::string& Filename)
{
    m_EntriesVertex.resize(pScene->mNumMeshes);
    m_EntriesIndices.resize(pScene->mNumMeshes);
    m_Textures.resize(pScene->mNumMaterials);
    m_TexturesName.resize(pScene->mNumMaterials);
    cudaTexture.resize(pScene->mNumMaterials);
    materialIndex.resize(pScene->mNumMeshes);
    meshEntriName.resize(pScene->mNumMeshes);
    // Initialize the meshes in the scene one by one
    for (unsigned int i = 0 ; i < m_EntriesVertex.size() ; i++) {
        const aiMesh* paiMesh = pScene->mMeshes[i];
        InitMesh(i, paiMesh);
    }
    return InitMaterials(pScene, Filename);
}

void MeshLoader::InitMesh(unsigned int Index, const aiMesh* paiMesh)
{
    materialIndex[Index] = paiMesh->mMaterialIndex;
    const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
    meshEntriName[Index] = paiMesh->mName.C_Str();
    for (unsigned int i = 0 ; i < paiMesh->mNumVertices ; i++) {
        const aiVector3D* pPos      = &(paiMesh->mVertices[i]);
        const aiVector3D* pNormal   = &(paiMesh->mNormals[i]);
        const aiVector3D* pTexCoord = paiMesh->HasTextureCoords(0) ? &(paiMesh->mTextureCoords[0][i]) : &Zero3D;

        Vertex v(vec3f(reinterpret_cast<const float&>(pPos->x), reinterpret_cast<const float&>(pPos->y), reinterpret_cast<const float&>(pPos->z)),
                 vec2f(pTexCoord->x,pTexCoord->y),
                 vec3f(pNormal->x, pNormal->y, pNormal->z));
        m_EntriesVertex[Index].push_back(v);
    }

    for (unsigned int i = 0 ; i < paiMesh->mNumFaces ; i++) {
        const aiFace& Face = paiMesh->mFaces[i];
        if (Face.mNumIndices < 3) {
            continue;
        }
        m_EntriesIndices[Index].push_back(vec3i(Face.mIndices[0],Face.mIndices[1],Face.mIndices[2]));
   }
}

bool MeshLoader::InitMaterials(const aiScene* pScene, const std::string& Filename)
{
    // Extract the directory part from the file name
    std::string::size_type SlashIndex = Filename.find_last_of("/");
    std::string Dir;

    if (SlashIndex == std::string::npos) {
        Dir = ".";
    }
    else if (SlashIndex == 0) {
        Dir = "/";
    }
    else {
        Dir = Filename.substr(0, SlashIndex);
    }

    bool Ret = true;
    // Initialize the materials
    for (unsigned int i = 0 ; i < pScene->mNumMaterials ; i++) {
        const aiMaterial* pMaterial = pScene->mMaterials[i];
        if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString Path;
            if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
                std::string path = Path.data;
                std::replace(path.begin(), path.end(),'\\','/');
                std::string FullPath = Dir + "/" + path;

                image_t *img = new image_t();
                
                img->loadImage(FullPath);
                m_Textures[i] = img;
                m_TexturesName[i] = FullPath;
            }
            else {
                image_t *white = new image_t();
                white->createWhiteImage(100,100);
                m_Textures[i] = white;
            }
        }
        else 
        {
            image_t *white = new image_t();
            white->createWhiteImage(100,100);
            m_Textures[i] = white;
        }
        createTexture(i);
    }
    return Ret;
}

std::vector<TriangleMesh*> MeshLoader::toTriangleMeshs(){
    std::vector<TriangleMesh*> triangle;
    triangle.reserve(m_EntriesVertex.size());
    std::vector<vec3f> vertices;
    std::vector<vec2f> texCoord;

    vec3f min = vec3f(9999999999999.f), max= vec3f(-99999999999999999.f);

    for(int i = 0; i < m_EntriesVertex.size(); ++i)
        for(int j  = 0; j <m_EntriesVertex[i].size() ; ++j){
            min = minVec(min,m_EntriesVertex[i][j].m_pos);
            max = maxVec(max,m_EntriesVertex[i][j].m_pos);
        }
    const vec3f d = max - min;
    const vec3f c = min + d/2.0f;
    for(int i = 0; i < m_EntriesVertex.size(); ++i){
       vertices.resize(m_EntriesVertex[i].size());
        texCoord.resize(m_EntriesVertex[i].size());

        for(size_t j = 0; j < m_EntriesVertex[i].size(); ++j){
            vertices[j] = ( m_EntriesVertex[i][j].m_pos ) -  (c); // conversion pour remettre entre [0,1]
            texCoord[j] = m_EntriesVertex[i][j].m_tex;
        }
        TriangleMesh* t = new TriangleMesh();

        t->addVertices(vertices);
        t->addIndices(m_EntriesIndices[i]);
        t->addTextureCoordinate(texCoord);
        t->setTexture(cudaTexture[this->materialIndex[i]]);
        triangle.push_back(t);
    }

    return triangle;
}

void MeshLoader::createTexture(int index){
    uint8_t* h_array;
    vec3i pixelSize(m_Textures[index]->width,m_Textures[index]->height,1);
    int uCount = pixelSize.x * pixelSize.y;
 
    h_array = (uint8_t*)malloc(uCount * sizeof(uint8_t) * 4);
    for(int i = 0; i < uCount ; i++){
        const int x = i % m_Textures[index]->width;
        const int y = i / m_Textures[index]->width;
        int r = 0,g = 0,b = 0;

        h_array[i*4]   = m_Textures[index]->data[i*4];
        h_array[i*4+1] = m_Textures[index]->data[i*4+1];
        h_array[i*4+2] = m_Textures[index]->data[i*4+2];
        h_array[i*4+3] = 255;
    }

    cudaResourceDesc res_desc = {};

     cudaChannelFormatDesc channel_desc;
     int32_t width  = m_Textures[index]->width;
     int32_t height = m_Textures[index]->height;
     int32_t numComponents = 4;
     int32_t pitch  = width*numComponents*sizeof(uint8_t);
     channel_desc = cudaCreateChannelDesc<uchar4>();

     cudaArray_t   &pixelArray = d_array;
     CUDA_CHECK(MallocArray(&pixelArray,
                            &channel_desc,
                            width,height));

     CUDA_CHECK(Memcpy2DToArray(pixelArray,
                                /* offset */0,0,
                                h_array,
                                pitch,pitch,height,
                                cudaMemcpyHostToDevice));

     res_desc.resType          = cudaResourceTypeArray;
     res_desc.res.array.array  = pixelArray;

     cudaTextureDesc tex_desc     = {};
     tex_desc.addressMode[0]      = cudaAddressModeWrap;
     tex_desc.addressMode[1]      = cudaAddressModeWrap;
     tex_desc.filterMode          = cudaFilterModeLinear;
     tex_desc.readMode            = cudaReadModeNormalizedFloat;
     tex_desc.normalizedCoords    = 1;
     tex_desc.maxAnisotropy       = 1;
     tex_desc.maxMipmapLevelClamp = 99;
     tex_desc.minMipmapLevelClamp = 0;
     tex_desc.mipmapFilterMode    = cudaFilterModePoint;
     tex_desc.borderColor[0]      = 1.0f;
     tex_desc.sRGB                = 0;

     // Create texture object
     CUDA_CHECK(CreateTextureObject(&cudaTexture[index], &res_desc, &tex_desc, nullptr));

    free(h_array);
}
