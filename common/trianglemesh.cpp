#include "trianglemesh.h"


TriangleMesh::TriangleMesh()
{
    index.resize(0);
    vertex.resize(0);
    texCoordBuffer.resize(0);
    color = vec3f(1.f,0.f,0.f);
}

TriangleMesh::~TriangleMesh(){

    vertexBuffer.free();
    indexBuffer.free();
    texCoordBuffer.free();
    
    vertex.clear();
    index.clear();
    texCoord.clear();
}

void TriangleMesh::addPlane(vec3f &center, vec3f &size, vec3f &color){
    this->center = center;
    this->size = size;
    this->color = color;
    int firstVertexID = (int)vertex.size();
    vertex.push_back((vec3f(-0.5f, -0.0f, -0.5f)* size + center) );
    vertex.push_back((vec3f(0.5f, -0.0f, -0.5f)* size+ center) );
    vertex.push_back((vec3f(0.5f, 0.0f, 0.5f)* size+ center) );
    vertex.push_back((vec3f(-0.5f, 0.0f, 0.5f)* size+ center) );
    int indices[] = { 0,1,2, 2,3,0};
    for (int i = 0; i < 2; i++)
        index.push_back( vec3i(indices[3 * i + 0] + firstVertexID,
            indices[3 * i + 1],
            indices[3 * i + 2]));
}

void TriangleMesh::addUnitCube(){
    int firstVertexID = (int)vertex.size();
    vertex.push_back(vec3f(-1.f, -1.f, -1.f));
    vertex.push_back(vec3f(1.f, -1.f, -1.f));
    vertex.push_back(vec3f(-1.f, 1.f, -1.f));
    vertex.push_back(vec3f(1.f, 1.f, -1.f));
    vertex.push_back(vec3f(-1.f, -1.f, 1.f));
    vertex.push_back(vec3f(1.f, -1.f, 1.f));
    vertex.push_back(vec3f(-1.f, 1.f, 1.f));
    vertex.push_back(vec3f(1.f, 1.f, 1.f));


    int indices[] = { 0,1,3, 2,3,0,
                     5,7,6, 5,6,4,
                     0,4,5, 0,5,1,
                     2,3,7, 2,7,6,
                     1,5,7, 1,7,3,
                     4,0,2, 4,2,6
    };
    for (int i = 0; i < 12; i++)
        index.push_back( vec3i(indices[3 * i + 0] + firstVertexID,
            indices[3 * i + 1],
            indices[3 * i + 2]));
}


void TriangleMesh::getSBT(sbtData *sbt){
    TriangleMeshSBT& sbtTriangle = sbt->meshData;
    sbtTriangle.kd = color;
    sbtTriangle.vertex = reinterpret_cast<vec3f*>(this->vertexBuffer.d_ptr);
    sbtTriangle.indices = reinterpret_cast<vec3i*>(this->indexBuffer.d_ptr);
    sbtTriangle.texCoord = reinterpret_cast<vec2f*>(this->getTextureCoordinateBuffer());
    sbtTriangle.tex = this->tex;
    sbtTriangle.hasTexture = (this->tex == 0 ? 0 : 1);
}
void TriangleMesh::setTexture(cudaTextureObject_t texture){
    tex = texture;
}
CUdeviceptr TriangleMesh::getTextureCoordinateBuffer() {
    if( texCoord.size() != 0){
        texCoordBuffer.free();
        texCoordBuffer.alloc_and_upload(texCoord);
        return texCoordBuffer.d_pointer();
    }else{
        return 0;
    }
}
size_t TriangleMesh::getNumVertices() const{
    return vertex.size();
}

CUdeviceptr TriangleMesh::getVertexDevicePointer() {
    vertexBuffer.free();
    vertexBuffer.alloc_and_upload(vertex);
    return vertexBuffer.d_pointer();
}


CUdeviceptr TriangleMesh::getIndexDevicePointer() {
    indexBuffer.free();
    indexBuffer.alloc_and_upload(index);
    return indexBuffer.d_pointer();
}

size_t TriangleMesh::getNumIndex() const {
    return index.size();
}


void TriangleMesh::addVertices(const std::vector<vec3f> vertices){
    for(size_t i =0; i < vertices.size() ; ++i){
        this->vertex.push_back(vertices[i]);
    }
}

void TriangleMesh::addIndices(const std::vector<vec3i> indices){
    for(size_t i =0; i < indices.size() ; ++i){
        this->index.push_back(indices[i]);
    }
}
void TriangleMesh::addTextureCoordinate(const std::vector<vec2f> texCoords){
    for(size_t i =0; i < texCoords.size() ; ++i){
        this->texCoord.push_back(texCoords[i]);
    }
    texCoordBuffer.free();
    texCoordBuffer.alloc_and_upload(texCoord);
}
void TriangleMesh::translate(const vec3f &t){
    for(size_t i = 0; i < vertex.size(); ++i){
           vertex[i] = vertex[i] + t;
    }
}

void TriangleMesh::resize(const vec3f &newDim){
    vec3f min = vec3f(9999999999999.0f),max=vec3f(-9999999999999999.0f);
    for(size_t i = 0; i < vertex.size(); ++i){
        min = minVec(min,vertex[i]);
        max = maxVec(max,vertex[i]);
    }
    for(size_t i = 0; i < vertex.size(); ++i){
        vertex[i] = ((vertex[i]-min)/(max-min)); // compris entre [0;1]
        vertex[i] = (vertex[i] - 0.5f) * newDim;
    }
}

vec3f TriangleMesh::getCenter(){
    vec3f c = vec3f(0);

    for(int i = 0; i < vertex.size(); ++i )
        c = c + vertex[i];
    return c / (float)(vertex.size());
}

vec3f TriangleMesh::getSize(){
    vec3f min = vec3f(9999999999999.0f),max=vec3f(-9999999999999999.0f);
    for(size_t i = 0; i < vertex.size(); ++i){
        min = minVec(min,vertex[i]);
        max = maxVec(max,vertex[i]);
    }
    return max - min;
}


void TriangleMesh::setColor(vec3f &c){
    color = c;
}
