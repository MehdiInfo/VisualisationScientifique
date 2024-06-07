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
#define stepSize_current optixLaunchParams.frame.sampler

#include <optix_device.h>
#include "../common/LaunchParams.h"

  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;
  
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

  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = vec3f(0.f,0.f,0.f);
    // set to constant white as background color

  }

  extern "C" __global__ void __raygen__renderFrame(){
     // compute a test pattern based on pixel ID
     // width [0 , 7680]
     // height [0,4320]
     const int ix = optixGetLaunchIndex().x;
     const int iy = optixGetLaunchIndex().y;

     const auto &camera = optixLaunchParams.camera;

     // our per-ray data for this example. what we initialize it to
     // won't matter, since this value will be overwritten by either

     // the miss or hit program, anyway
     vec3f pixelColorPRD = vec3f(0.f);

     // the values we store the PRD pointer in:
     uint32_t u0, u1;
     packPointer( &pixelColorPRD, u0, u1 );

     // normalized screen plane position, in [0,1]^2

     const vec2f size = vec2f(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
     const vec2f origin_uv = (vec2f((float)(ix+0.5f),(float)(iy+0.5f)) /
                                        vec2f((float)optixLaunchParams.frame.size.x,(float)optixLaunchParams.frame.size.y));

     pixelColorPRD = vec3f(0.f);
     const vec3f rayDir = normalize(camera.direction
                              + (origin_uv.x - 0.5f)  * camera.horizontal
                              + (origin_uv.y - 0.5f) * camera.vertical);

     const vec3f ro = camera.position;
     const vec3f rd = rayDir;

     const float3 cp = make_float3(ro.x ,ro.y,ro.z);
     const float3 rdf3 = make_float3(rd.x,rd.y,rd.z);


     optixTrace(optixLaunchParams.traversable,
                    cp,
                    rdf3,
                    0.f,    // tmin
                    100.f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_NONE,
                    SURFACE_RAY_TYPE,             // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    SURFACE_RAY_TYPE,             // missSBTIndex
                    u0, u1 );
     const unsigned char r = int(255.99f*pixelColorPRD.x);
     const unsigned char g = int(255.99f*pixelColorPRD.y);
     const unsigned char b = int(255.99f*pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
       | (r<<0) | (g<<8) | (b<<16);

    const uint32_t fbIndex = (ix)+iy*(optixLaunchParams.frame.size.x);

    uint32_t &pixel = optixLaunchParams.frame.colorBuffer[fbIndex];
    pixel = (pixel & 0xFFFFFF00) | ((uint32_t)r <<  0);
    pixel = (pixel & 0xFFFF00FF) | ((uint32_t)g <<  8);
    pixel = (pixel & 0xFF00FFFF) | ((uint32_t)b << 16);
    pixel = (pixel & 0x00FFFFFF) | ((uint32_t)255 << 24);
     
 }

