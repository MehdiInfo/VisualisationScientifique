#ifndef OPTIXVIEWER_H
#define OPTIXVIEWER_H

#include "camera.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "vec.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <cuda_gl_interop.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include "Scene.hpp"

/**
    OptixRender est notre ray tracer. Cette objet nous permet de créer un context Optix,
    de l'initialiser et d'y charger les différentes informations de nos objets.

    Il est composé de nombreuses fonctions. Cependant, il gère seulement les échangent de nos données entre notre scèene et notre contexte Optix.
    Notre contexte optix est composé de différentes parties :
        -VAS : Volume Acceleration Structures
        -MAS : Mesh Acceleration Strcuture
        -IAS : Instance Acceleration Structure
        -Différent Module.
        -SBT : Shading Binding Table.
    Le VAS nous permet de gérer l'ensemble des informations qui non pas de maillage. Il gère donc l'ensemble des formes implicites et nos volumes.
    Le MAS nous permet de gérer les maillages triangulaires.
    L'IAS nous permet de gérer les intersections des VAS et MAS en un seul lancé de rayon. Il représente la "fusion" du VAS et MAS.
    Les modules représentent l'ensemble des .cu. Les .cu sont constituées de différent programmes :
    (Closest Hit program, Any Hit program, InterSection program). Ces programmes nous permettent de gérer le rendu et l'ensemble des itnersections.

    Le SBT nous permet de gérer les données propres aux objets de notre scènes, par exemple, le centre de notre volume.
*/

class OptixRender
{
public:
    OptixRender();
    OptixRender(Scene *modele);

    ~OptixRender();
    LaunchParams* getLaunchParams();
    /**
        \brief Initialize le contexte OptiX, charge les différents modules pour réaliser les rendus.
        Créer les VAS, MAS et IAS pour pouvoir identifier les intersections et instancie la SBT.
    */
    void initialize(Scene *modele);

    /**
        \brief Réalise le rendu d'une image
    */
    void render();

    /**
        Dimensionne la taille de la fenêtre de rendu
    */
    void resize(const vec2i &newSize);

    /**
        \brief Permet de récupérer les données de l'image qui a été rendu
    */
    void downloadPixels(uint32_t h_pixels[]);

    void setCamera(const Camera &cam);
    /**
        Permet de fixer la cameré OptiX.
        La cameré passé en paramètre à les propriétés de la caméra OpenGL pour avoir une superposition correcte.
    */

    void updateSBTBuffer();

    /* helper function that initializes optix and checks for errors */
    void initOptix();

    /* creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /* creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModules();
    void createVolumeModule();

    /* does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();

    /* does all setup for the miss program(s) we are going to use */
    void createMissPrograms();

    /* does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();
    void createVolumeHitgroupPrograms();
   
    /* assembles the full pipeline of all programs */
    void createPipeline();

    /* constructs the shader binding table */
    void buildSBT();

    /* build an acceleration structure for the given triangle mesh */
    void buildVolumeAccel();

    void updateAccelerationStructure();

    void updateSBT();
    //Mets à jours les données concernants les aabb structures
    void updateVAS();

    /*Permet de demander la mise à jours du VAS,MAS  IAS ou de la SBT
        Ceci, permet d'éviter de régénérer une instance au besoin car la génération des  GAS est lourdes.
    */
    void notifyMeshChanges();
    void notifySbtChanges();
    void notifyAabbObjectChanges();

private :

    /* SBT record for a raygen program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here
      void *data;
    };

    // SBT record for a miss program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here
      void* data;
    };

    /* SBT record for a hitgroup program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here
      sbtData sbt = {};
    };
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupMeshRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here

    };
  protected:
    /* @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /* @} */

    // the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /* @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};
    /* @} */

    /* @{ the module that contains out device programs */
    OptixModule                 volume_module;
    OptixModule                 raygen_module;

    OptixModuleCompileOptions   moduleCompileOptions = {};
    /* @} */

    /* vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};
    CUDABuffer hitgroupMeshRecordsBuffer;
    OptixShaderBindingTable sbtMesh = {};

    /* @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /* @} */

    CUDABuffer colorBuffer;

    CUDABuffer vasBuffer;
    OptixTraversableHandle vas{0};

    Scene* modele;
    bool isVolumeDataModified = false;
    bool isMeshDataModified = false;
    bool isSBTDataModified = false;

    std::string ptx_volume_path = "./CMakeFiles/volume.dir/shader/volume.ptx";;
    std::string ptx_raygen_path = "./CMakeFiles/raygen.dir/shader/raygen.ptx";

    std::vector<HitgroupRecord> hitgroupRecords;

    //FOR VAS
    OptixAabb *aabb;
    CUdeviceptr d_aabb;
    CUDABuffer vasTempBuffer;
    CUDABuffer vasOutputBuffer;
};

#endif // OPTIXVIEWER_H
