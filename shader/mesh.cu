// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define DEVICE 1

#include <optix_device.h>

#include "../common/LaunchParams.h"

  __device__ void swap(float &a, float &b) {
      float tmp = a;
      a = b;
      b = tmp;
  }
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  __device__ vec3i pi2Index(vec3f pi, vec3f size, vec3f center, vec3i sizePixel) {
      const vec3f findex = ((pi - center + size / 2) / size);
      vec3i index = vec3i((int)(findex.x) * sizePixel.x, (int)(findex.y) * sizePixel.y, (int)(findex.z) * sizePixel.z);
      return index;
  }

  extern "C" __global__ void __closesthit__mesh_radiance(){
      const sbtData &sbt = (*reinterpret_cast<const sbtData*>(optixGetSbtDataPointer()));
        const TriangleMeshSBT& sbt_data =  sbt.meshData;

        vec3f& prd = *(vec3f*)getPRD<vec3f>();
        //Gather informations
        const vec3f ro = optixGetWorldRayOrigin();
        const vec3f rd = optixGetWorldRayDirection();
        const vec3f sizeP = vec3f(1.0f);
        const float fov = acos(0.6f);
        const vec3f Li = vec3f(1.f);
        //Ray
        const int   primID = optixGetPrimitiveIndex();
        const vec3i index  = sbt_data.indices[primID];
        const vec3f L      =  normalize((sbt_data.vertex[index.x],sbt_data.vertex[index.y],sbt_data.vertex[index.z])/3.0f - ro);
        const vec3f &A     = sbt_data.vertex[index.x];
        const vec3f &B     = sbt_data.vertex[index.y];
        const vec3f &C     = sbt_data.vertex[index.z];
        const vec3f N     = normalize(cross(B-A,C-A));
        float4 color = make_float4(1.0f,1.0f,1.0f,1.0f);
        if( sbt_data.hasTexture != 0){
            const float u = optixGetTriangleBarycentrics().x;
            const float v = optixGetTriangleBarycentrics().y;
            const vec2f tc
                    = (1.f-u-v)* sbt_data.texCoord[index.x]
                    +         u * sbt_data.texCoord[index.y]
                    +         v * sbt_data.texCoord[index.z];
            
            color = tex2D<float4>(sbt_data.tex,tc.x,tc.y);

      }
      prd = Li * vec3f(color.x,color.y,color.z)  *  fabs(dot(L,N));
  }

  extern "C" __global__ void __anyhit__mesh_radiance(){

  }

