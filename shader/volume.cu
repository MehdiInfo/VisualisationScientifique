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
#define nbSamples optixLaunchParams.frame.sampler

#include <optix_device.h>
#include <optix.h>
#include "../common/LaunchParams.h"
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  extern "C" int mode;
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
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------

  __device__ void swap(float &a, float &b) {
    float tmp = a;
    a = b;
    b = tmp;
  }
  __device__ bool inVolume(const VolumetricCube &data, const vec3f &pi){
        bool res = false;
        const vec3f min = data.center - data.size/2.0f;
        const vec3f max = data.center + data.size/2.0f;
        if(( pi.x <= max.x && pi.x >= min.x)&&
                ( pi.y <= max.y && pi.y >= min.y) &&
                ( pi.z <= max.z && pi.z >= min.z))
            res = true;
        return res;
  }
  
  __device__ void mip(){
      const VolumetricCube& data = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
     const int   primID = optixGetPrimitiveIndex();
     intersection_time time;
     time.tmin.uitmin = optixGetAttribute_0();
     time.tmax.uitmax = optixGetAttribute_1();
  
     vec3f ro = optixGetWorldRayOrigin();
     vec3f rd = optixGetWorldRayDirection();
     vec3f& prd = *(vec3f*)getPRD<vec3f>();
     vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);

     //Ray
     vec3f point_in = ro + time.tmin.ftmin * rd ;
     vec3f point_out = ro + time.tmax.ftmax * rd;
     vec3f ray_world = point_out - point_in;

     const float stepSize_current = norme(point_out - point_in) / nbSamples;
     vec3f step_vector_tex = normalize(ray_world) * stepSize_current;
     float current_ray_length = norme(ray_world);

     vec3f current_pos_tex = point_in;
     float current_max = 0.0f;
     float current_intensity = 0.0f;

     //MIP
     prd = vec3f(0.0f);
     while(current_ray_length > 0.0f){
        vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;
        current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);


        if( current_intensity >= optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity){
            if( current_intensity > current_max )
                current_max = current_intensity;

            if( current_max == 1.0f){
                prd = vec3f(1.0f);
                break;
            }
            prd = vec3f(current_max);
        }
        current_pos_tex = current_pos_tex + step_vector_tex;
        current_ray_length -= stepSize_current;
     }
  }
  __device__ void Mean() {
      //
      const VolumetricCube& data = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
      const int   primID = optixGetPrimitiveIndex();
      intersection_time time;
      time.tmin.uitmin = optixGetAttribute_0();
      time.tmax.uitmax = optixGetAttribute_1();
   
      vec3f ro = optixGetWorldRayOrigin();
      vec3f rd = optixGetWorldRayDirection();
      vec3f& prd = *(vec3f*)getPRD<vec3f>();
      vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);
      //
      float intensity_acc = 0.0f;
      float current_intensity = 0.0f;

      
      vec3f point_in = ro + time.tmin.ftmin * rd;

      vec3f point_out = ro + time.tmax.ftmax * rd;
      vec3f ray_world = point_out - point_in;
      const float stepSize_current = norme(point_out - point_in) / nbSamples;
      vec3f step_vector_tex = normalize(ray_world) * stepSize_current;

      
      vec3f current_pos_tex = point_in;
      float current_ray_length = norme(ray_world);

      while(current_ray_length > 0.0f ) {//
          vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;//

          current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);//

          if(current_intensity >= optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity) {
              intensity_acc += current_intensity;
              prd = vec3f(intensity_acc);
          }

          current_pos_tex = current_pos_tex + step_vector_tex;
          current_ray_length -= stepSize_current;
      }
      

  }

  __device__ void MinIP(){
      //
      const VolumetricCube& data = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
      const int   primID = optixGetPrimitiveIndex();
      intersection_time time;
      time.tmin.uitmin = optixGetAttribute_0();
      time.tmax.uitmax = optixGetAttribute_1();
   
      vec3f ro = optixGetWorldRayOrigin();
      vec3f rd = optixGetWorldRayDirection();
      vec3f& prd = *(vec3f*)getPRD<vec3f>();
      vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);
      //
      float minimum_intensity = 1.1f;
      float current_intensity = 0.0f;

      
      vec3f point_in = ro + time.tmin.ftmin * rd;

      vec3f point_out = ro + time.tmax.ftmax * rd;
      vec3f ray_world = point_out - point_in;
      const float stepSize_current = norme(point_out - point_in) / nbSamples;
      vec3f step_vector_tex = normalize(ray_world) * stepSize_current;

      
      vec3f current_pos_tex = point_in;
      float current_ray_length = norme(ray_world);

      while (current_ray_length > 0.0f ) {
          vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;
          
          current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);
          
          if (current_intensity >= optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity && current_intensity < minimum_intensity) {
              minimum_intensity = current_intensity;
              prd = vec3f(minimum_intensity);
          }
          
          current_pos_tex = current_pos_tex + step_vector_tex;
          current_ray_length -= stepSize_current;
      }
      
  }

  __device__ void LMIP() {
    //
    const VolumetricCube& data = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
    const int   primID = optixGetPrimitiveIndex();
    intersection_time time;
    time.tmin.uitmin = optixGetAttribute_0();
    time.tmax.uitmax = optixGetAttribute_1();
 
    vec3f ro = optixGetWorldRayOrigin();
    vec3f rd = optixGetWorldRayDirection();
    vec3f& prd = *(vec3f*)getPRD<vec3f>();
    vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);
    //
    int threshold_reached = 0;
    float previous_intensity = -1.0f;
    float current_intensity = 0.0f;
    vec3f previous_location_tex = make_float3(-1.0f, -1.0f, -1.0f);

    
    vec3f point_in = ro + time.tmin.ftmin * rd;

    vec3f point_out = ro + time.tmax.ftmax * rd;
    vec3f ray_world = point_out - point_in;
    const float stepSize_current = norme(point_out - point_in) / nbSamples;
    vec3f step_vector_tex = normalize(ray_world) * stepSize_current;

    
    vec3f current_pos_tex = point_in;
    float current_ray_length = norme(ray_world);

    while (current_ray_length > 0.0f ) {
        vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;

        current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);

        if (threshold_reached == 0 && current_intensity >= optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity) {
            threshold_reached = 1;
        }

        if (threshold_reached == 1) {
            if (current_intensity <= previous_intensity) {
                break;
            } else {
                previous_intensity = current_intensity;
                previous_location_tex = current_pos_tex;
            }
            prd = vec3f(previous_intensity);
        }

        current_pos_tex = current_pos_tex + step_vector_tex;
        current_ray_length -= stepSize_current;
    }
    
  }

  __device__ void DEMIP() {
      //
      const VolumetricCube& data = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
      const int   primID = optixGetPrimitiveIndex();
      intersection_time time;
      time.tmin.uitmin = optixGetAttribute_0();
      time.tmax.uitmax = optixGetAttribute_1();
   
      vec3f ro = optixGetWorldRayOrigin();
      vec3f rd = optixGetWorldRayDirection();
      vec3f& prd = *(vec3f*)getPRD<vec3f>();
      vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);
      //
      float maximum_intensity = 0.0f;
      float current_intensity = 0.0f;
      vec3f maximum_intensity_location_tex = make_float3(-1.0f, -1.0f, -1.0f);
      vec3f maximum_intensity_location_world = make_float3(-1.0f, -1.0f, -1.0f);

      
      vec3f point_in = ro + time.tmin.ftmin * rd;

      vec3f point_out = ro + time.tmax.ftmax * rd;
      vec3f ray_world = point_out - point_in;
      const float stepSize_current = norme(point_out - point_in) / nbSamples;
      vec3f step_vector_world = normalize(ray_world) * stepSize_current;
      vec3f step_vector_tex = normalize(ray_world) * stepSize_current;

      
      vec3f current_pos_tex = point_in;
      float current_ray_length = norme(ray_world);

      while (current_ray_length > 0.0f ) {
            vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;

          current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);

          if (current_intensity >= optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity && current_intensity > maximum_intensity) {
              maximum_intensity = current_intensity;
              maximum_intensity_location_tex = current_pos_tex;
              prd = vec3f(maximum_intensity);
          }

          if (maximum_intensity == 1.0f){
            prd = vec3f(1.0f);
            break;
          } 

          current_pos_tex = current_pos_tex + step_vector_tex;
          current_ray_length -= stepSize_current;
      }
      
      
  }

  extern "C" __global__ void __closesthit__volume_radiance(){
      const VolumetricCube& data
       = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
      // Change le mode de Ray Tracing en fonction du type séléctioner sur l'interface graphique
      switch(optixLaunchParams.frame.renderType) {
        case 0:
          Mean();
          break;
        case 1:
          MinIP();
          break;
        case 2:
          LMIP();
          break;
        case 3:
          DEMIP();
          break;
        case 4:
          mip();
          break;
        default:
          mip();
      }
  }


  extern "C" __global__ void __anyhit__volume_radiance()
  {
  }


  extern "C" __global__ void __intersection__volume() {
      const VolumetricCube& sbtData
          = *(const VolumetricCube*)optixGetSbtDataPointer();
      vec3f ro = optixGetWorldRayOrigin();
      vec3f rayDir = optixGetWorldRayDirection();
      vec3f min, max;
      min = sbtData.center - sbtData.size / 2;
      max = sbtData.center + sbtData.size / 2;

      float tmin = (min.x - ro.x) / rayDir.x;
      float tmax = (max.x - ro.x) / rayDir.x;

      if (tmin > tmax) swap(tmin, tmax);

      float tymin = (min.y - ro.y) / rayDir.y;
      float tymax = (max.y - ro.y) / rayDir.y;

      if (tymin > tymax) swap(tymin, tymax);

      //Rayon en dehors du cube (normalement impossible )
      if ((tmin > tymax) || (tymin > tmax))

        return ;

      if (tymin > tmin)
          tmin = tymin;

      if (tymax < tmax)
          tmax = tymax;

      float tzmin = (min.z - ro.z) / rayDir.z;
      float tzmax = (max.z - ro.z) / rayDir.z;

      if (tzmin > tzmax) swap(tzmin, tzmax);

      //Rayon en dehors du cube (normalement impossible )
      if ((tmin > tzmax) || (tzmin > tmax))
        return ;

      if (tzmin > tmin)
          tmin = tzmin;

      if (tzmax < tmax)
          tmax = tzmax;
      if (tmin > tmax) swap(tmin, tmax);
      intersection_time time;

      time.tmin.ftmin = tmin;
      time.tmax.ftmax = tmax;
      optixReportIntersection(tmin, 1,time.tmin.uitmin, time.tmax.uitmax);
  }
