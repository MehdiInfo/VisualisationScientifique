#include "volume.h"

Volume::Volume()
{
    worldSize = vec3f(1.0f);
    pixelSize = vec3i(100,100,100);
    position = vec3f(0.f);
    minIntensity = 4096;
    maxIntensity = 0;
}

Volume::Volume(std::string path){
    loadVolume(path);
    createTexture();
}


void Volume::loadVolume(const std::string &path){
     FILE* fp = fopen(path.c_str(), "rb");
    unsigned short vuSize[3];
    fread((void*)vuSize, 3, sizeof(unsigned short), fp);
    pixelSize = vec3i(int(vuSize[0]),int(vuSize[1]),int(vuSize[2]));
    unsigned int uCount = int(vuSize[0]) * int(vuSize[1]) * int(vuSize[2]);
    unsigned short *data = new unsigned short[uCount];
    fread((void*)data, uCount, sizeof(unsigned short), fp);
    fclose(fp);
    for(unsigned int i = 0; i < uCount; ++i){
        if( data[i] > maxIntensity)
            maxIntensity = data[i];
        if( data[i] < minIntensity)
            minIntensity = data[i];
    }
    pData = data;
    type = SHORT;

    createTexture();

    /*Set Dimension*/
    vec3f size = worldSize;
    size = vec3f(size.x * (float)pixelSize.x,size.y * (float)pixelSize.y,size.z * (float)pixelSize.z );
    float max = size.x;
    if( max < size.y )
        max = size.y;
    if( max < size.z)
        max = size.z;
    worldSize = size / max * worldSize;

}


void Volume::createTexture(){
    float* h_array;

    const unsigned int uCount = pixelSize.x * pixelSize.y * pixelSize.z;

    h_array = (float*)malloc(uCount * sizeof(float));

    for(unsigned int i = 0; i < uCount; ++i){
        switch(type){
            case SHORT :
                h_array[i] = (float)(((unsigned short*)(pData))[i]) / 4096.0f;
            break;
            case UCHAR:
                h_array[i] = (float)(((unsigned char*)(pData))[i]) / 255.0f;
            break;
        }
    }

    cudaChannelFormatDesc channel_descriptor = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(pixelSize.x,pixelSize.y,pixelSize.z);

    cudaMalloc3DArray(&d_array,&channel_descriptor,volumeSize);
    CUDA_SYNC_CHECK();
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_array, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_array;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&copyParams);
    CUDA_SYNC_CHECK();
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = d_array;

    texDesc.addressMode[0]      = cudaAddressModeWrap;
    texDesc.addressMode[1]      = cudaAddressModeWrap;
    texDesc.filterMode          = cudaFilterModeLinear;
    texDesc.readMode            = cudaReadModeElementType;
    texDesc.normalizedCoords    = 1;
    texDesc.maxAnisotropy       = 1;
    texDesc.maxMipmapLevelClamp = 99;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode    = cudaFilterModePoint;
    texDesc.borderColor[0]      = 1.0f;
    texDesc.sRGB                = 0;
    cudaCreateTextureObject(&cudaTexture,&resDesc,&texDesc,nullptr);
    CUDA_SYNC_CHECK();
    free(h_array);
}

cudaTextureObject_t Volume::getTextureReference() const{
    return cudaTexture;
}

void Volume::getAabb(float results[6]){
    OptixAabb* aabb = reinterpret_cast<OptixAabb*>(results);

   float3 m_min = make_float3(position.x - worldSize.x/2.f, position.y - worldSize.y/2.f, position.z - worldSize.z/2.f);
   float3 m_max = make_float3(position.x + worldSize.y/2.f, position.y + worldSize.y/2.f, position.z + worldSize.z/2.f);

   *aabb = {
       m_min.x, m_min.y, m_min.z,
       m_max.x, m_max.y, m_max.z
   };
}

void Volume::getSbt(sbtData *sbt){
    sbt->volumeData.tex = this->cudaTexture;
    sbt->volumeData.size = worldSize;
    sbt->volumeData.center = position;
    sbt->volumeData.sizePixel = pixelSize;

}

void Volume::translate(const vec3f &t){
    position = position + t;
}
vec3f Volume::getWorldSize() const {return worldSize;}
vec3f Volume::getWorldCenter() const { return position;}
vec3i Volume::getPixelSize() const {return pixelSize;}
CUdeviceptr Volume::getDevicePointer() const { return data.d_pointer();}
cudaArray_t Volume::getTextureDevicePointer() const { return d_array;}
CUDABuffer Volume::getCudaBuffer() const{return data;}

void Volume::uploadCudaBuffer(){
    switch(type){
        case SHORT :
            data.upload((unsigned short*)pData, pixelSize.x * pixelSize.y* pixelSize.z);
        break;
        case UCHAR:
            data.upload((unsigned char*)pData, pixelSize.x * pixelSize.y* pixelSize.z);
        break;
    }
}

void Volume::resizeCudaBuffer(){
    data.resize(pixelSize.x * pixelSize.y* pixelSize.z * sizeof(unsigned short));
}


void Volume::resize(const vec3f &newDim){
    worldSize = newDim;
}