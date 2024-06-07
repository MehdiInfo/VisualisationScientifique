#include "optixRender.h"

OptixRender::OptixRender(){}

OptixRender::OptixRender(Scene* modele)
{
    initialize(modele);
}



OptixRender::~OptixRender(){
      raygenRecordsBuffer.free();
      missRecordsBuffer.free();
      hitgroupRecordsBuffer.free();
      hitgroupMeshRecordsBuffer.free();
      launchParamsBuffer.free();

      iasBuffer.free();
      masBuffer.free();
      vasBuffer.free();

      free(aabb);
      vasTempBuffer.free();
      vasOutputBuffer.free();

      masTempBuffer.free();
      masOutputBuffer.free();

      iasTempBuffer.free();
      iasOutputBuffer.free();
}

void OptixRender::initialize(Scene *modele){
   initOptix();
   this->modele = modele;
   std::cout << "creating optix context ..." << std::endl;
   createContext();

   std::cout << "#osc: setting up module ..." << std::endl;
   createModules();

   std::cout << "#osc: creating raygen programs ..." << std::endl;
   createRaygenPrograms();
   std::cout << "#osc: creating miss programs ..." << std::endl;
   createMissPrograms();
   std::cout << "#osc: creating hitgroup programs ..." << std::endl;
   createHitgroupPrograms();


   if ( modele->getNumMesh()!= 0)buildMeshAccel();
   if ( modele->getNumVolume()!= 0)buildVolumeAccel();
   
   buildIAS(modele,vas, mas);
   launchParams.traversable = ias;
   std::cout << "#osc: setting up optix pipeline ..." << std::endl ;
   createPipeline();

   std::cout << "#osc: building SBT ..." << std::endl;
   buildSBT();

   launchParamsBuffer.alloc(sizeof(launchParams));
   std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;
}

LaunchParams* OptixRender::getLaunchParams(){return &launchParams;}

void OptixRender::buildIAS(Scene* modele, OptixTraversableHandle volume_traversable, OptixTraversableHandle mesh_traversable){
    std::vector<OptixInstance> instances;
    instances.resize(2);
    float transform[12];
    transform[0] = 1.0f; transform[1] = 0.0f; transform[2] = 0.0f; transform[3] = 0.0f;
    transform[4] = 0.0f; transform[5] = 1.0f; transform[6] = 0.0f; transform[7] = 0.0f;
    transform[8] = 0.0f; transform[9] = 0.0f; transform[10] = 1.0f; transform[11] = 0.0f;
    //instances 0 for volume
    size_t sbtOffset = 0;
    memcpy(instances[0].transform, transform, sizeof(float)*12);
    instances[0].instanceId = 0;
    instances[0].sbtOffset = (int)sbtOffset;
    instances[0].visibilityMask = 0xFF;
    instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
    instances[0].traversableHandle = volume_traversable;
    sbtOffset +=  modele->getNumVolume();

    //instance 1 for mesh
    memcpy(instances[1].transform, transform, sizeof(float)*12);
    instances[1].instanceId = 0;
    instances[1].sbtOffset = (int)sbtOffset;
    instances[1].visibilityMask = 0xFF;
    instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
    instances[1].traversableHandle = mesh_traversable;
    sbtOffset += modele->getNumMesh();

    CUDABuffer deviceInstances;
    deviceInstances.alloc_and_upload(instances);
    OptixBuildInput input;
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = deviceInstances.d_pointer();
    input.instanceArray.numInstances = (int)instances.size();
   // ==================================================================
   // BLAS setup
   // ==================================================================

   OptixAccelBuildOptions accelOptions;

   accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE
     | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
     ;
   accelOptions.motionOptions.numKeys  = 1;
   accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;


   OptixAccelBufferSizes blasBufferSizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage
               (optixContext,
                &accelOptions,
                &input,
                1,  // num_build_inputs
                &blasBufferSizes
                ));

   // ==================================================================
   // prepare compaction
   // ==================================================================

   CUDABuffer compactedSizeBuffer;
   compactedSizeBuffer.alloc(sizeof(uint64_t));

   OptixAccelEmitDesc emitDesc;
   emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emitDesc.result = compactedSizeBuffer.d_pointer();

   // ==================================================================
   // execute build (main stage)
   // ==================================================================

   iasTempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
   iasOutputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
   OPTIX_CHECK(optixAccelBuild(optixContext,
                               0,//stream
                               &accelOptions,
                               &input,
                               1,
                               iasTempBuffer.d_pointer(),
                               iasTempBuffer.sizeInBytes,

                               iasOutputBuffer.d_pointer(),
                               iasOutputBuffer.sizeInBytes,

                               &ias,

                               &emitDesc,1
                               ));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // perform compaction
   // ==================================================================
   uint64_t compactedSize;
   compactedSizeBuffer.download(&compactedSize,1);
   iasBuffer.alloc(compactedSize);
   OPTIX_CHECK(optixAccelCompact(optixContext,
                                 0,//stream
                                 ias,
                                 iasBuffer.d_pointer(),
                                 iasBuffer.sizeInBytes,
                                 &ias));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // aaaaaand .... clean up
   // ==================================================================
   compactedSizeBuffer.free();
   

}

void OptixRender::initOptix(){
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#OptiX: no CUDA capable devices found!");

    OPTIX_CHECK( optixInit() );
    cudaSetDevice(0);
}

void OptixRender::createContext(){
// for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS )
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
}

void OptixRender::createModules(){
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth          = 2;
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE | OPTIX_PRIMITIVE_TYPE_CUSTOM;

    std::cout << "Compile Volume module..." << std::endl;
    createVolumeModule();

    std::cout << "Compile Mesh module..." << std::endl;
    createMeshModule();

    std::cout << "All modules compiled !" << std::endl;
}

void OptixRender::createVolumeModule(){
    char log[2048];
    size_t sizeof_log = sizeof( log );
    std::ifstream inFile;
    inFile.open(ptx_volume_path); //open the input file
    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    std::string str = strStream.str(); //str holds the content of the file
    inFile.close();
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         str.c_str(),
                                         str.size(),
                                         log,&sizeof_log,
                                         &volume_module
                                         ));
}
void OptixRender::createMeshModule(){
    char log[2048];
    size_t sizeof_log = sizeof( log );

    std::ifstream inFile;
    inFile.open(ptx_mesh_path); //open the input file
    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    std::string str = strStream.str(); //str holds the content of the file
    inFile.close();
    
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         str.c_str(),
                                         str.size(),
                                         log,&sizeof_log,
                                         &mesh_module
                                         ));
}

/* does all setup for the raygen program(s) we are going to use */
void OptixRender::createRaygenPrograms(){
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    char log[2048];
    size_t sizeof_log = sizeof( log );
    std::ifstream inFile;
    inFile.open(ptx_raygen_path); //open the input file
    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    std::string str = strStream.str(); //str holds the content of the file
    inFile.close();

    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         str.c_str(),
                                         str.size(),
                                         log,&sizeof_log,
                                         &raygen_module
                                         ));

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = raygen_module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));

}

/*! does all setup for the miss program(s) we are going to use */
void OptixRender::createMissPrograms(){
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    //pgDesc.miss.module            = raygen_module;
    pgDesc.miss.module            = raygen_module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));

}

/*! does all setup for the hitgroup program(s) we are going to use */
void OptixRender::createHitgroupPrograms(){
    hitgroupPGs.resize(2);

    std::cout << "Create volumic hitgroup programs..." << std::endl;
    createVolumeHitgroupPrograms();

    std::cout << "Create mesh hitgroup programs..." << std::endl;
    createMeshHithtoupPrograms();
}

void OptixRender::createVolumeHitgroupPrograms(){
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = volume_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__volume_radiance";
    pgDesc.hitgroup.moduleAH            = volume_module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__volume_radiance";
    pgDesc.hitgroup.moduleIS = volume_module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__volume";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[VOLUME_RENDERING]
                                        ));

}

void OptixRender::createMeshHithtoupPrograms(){
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = mesh_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__mesh_radiance";
    pgDesc.hitgroup.moduleAH            = mesh_module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__mesh_radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[MESH_RENDERING]
                                        ));

}

/*! assembles the full pipeline of all programs */
void OptixRender::createPipeline(){
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));


    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));

}

/* constructs the shader binding table */
void OptixRender::buildSBT(){
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (size_t i=0;i<raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (size_t i=0;i<missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    for(size_t i = 0; i < modele->getNumVolume(); ++i){
        HitgroupRecord rec;
        memset(&rec.sbt,0,sizeof(sbtData));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[VOLUME_RENDERING],&rec));
        modele->getVolumeSbt(i,&rec.sbt);
        if( i == 0)
          rec.sbt.volumeData.color = vec3f(1.f,0.f,0.f);
        else 
          rec.sbt.volumeData.color = vec3f(0.f,1.f,0.f);
        hitgroupRecords.push_back(rec);
    }

    //Create HitGroup for mesh object
    const int numMeshObjects = (int)modele->getNumMesh();
    for (int i=0; i<numMeshObjects;i++) {
      HitgroupRecord rec;
      memset(&rec.sbt,0,sizeof(sbtData));
      modele->getMeshSbt(i,&rec.sbt);
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[MESH_RENDERING],&rec));
      hitgroupRecords.push_back(rec);
    }

    if( modele->getNumVolume() != 0 || modele->getNumMesh() != 0){
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
    }
}

void OptixRender::buildMeshAccel(){
    size_t nb_obj_mesh = modele->getNumMesh();
  // ==================================================================
  // triangle inputs
  // ==================================================================
  // upload the model to the device: the builder
  std::vector<OptixBuildInput> input;
  input.resize(nb_obj_mesh);
  std::vector<CUdeviceptr> d_vertices ;
  std::vector<CUdeviceptr> d_indices;
  d_vertices.resize(nb_obj_mesh);
  d_indices.resize(nb_obj_mesh);
  uint32_t triangleInputFlags[1] = { 0 };
  
  for(int i = 0; i < (int)nb_obj_mesh; ++i){
    input[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[i] = modele->getMesh(i)->getVertexDevicePointer();
    d_indices[i]  = modele->getMesh(i)->getIndexDevicePointer();

    input[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input[i].triangleArray.vertexStrideInBytes = sizeof(vec3f);
    input[i].triangleArray.numVertices = (int)modele->getMesh(i)->getNumVertices();
    input[i].triangleArray.vertexBuffers = &d_vertices[i];

    input[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input[i].triangleArray.indexStrideInBytes = sizeof(vec3i);
    input[i].triangleArray.numIndexTriplets = (int)modele->getMesh(i)->getNumIndex();
    input[i].triangleArray.indexBuffer = d_indices[i];

    // in this example we have one SBT entry, and no per-primitive
              // materials:
    input[i].triangleArray.flags = triangleInputFlags;
    input[i].triangleArray.numSbtRecords = 1;
    input[i].triangleArray.sbtIndexOffsetBuffer = 0;
    input[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
    input[i].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(sbtData);

  }
   // ==================================================================
   // BLAS setup
   // ==================================================================
   OptixAccelBuildOptions accelOptions;

       accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE
         | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
         ;
       accelOptions.motionOptions.numKeys  = 1;
       accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;


   OptixAccelBufferSizes blasBufferSizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage
               (optixContext,
                &accelOptions,
                input.data(),
                (int)input.size(),  // num_build_inputs
                &blasBufferSizes
                ));

   // ==================================================================
   // prepare compaction
   // ==================================================================

   CUDABuffer compactedSizeBuffer;
   compactedSizeBuffer.alloc(sizeof(uint64_t));

   OptixAccelEmitDesc emitDesc;
   emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emitDesc.result = compactedSizeBuffer.d_pointer();

   // ==================================================================
   // execute build (main stage)
   // ==================================================================

   masTempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
   masOutputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

   OPTIX_CHECK(optixAccelBuild(optixContext,
                               0, //stream
                               &accelOptions,
                               input.data(),
                               (int)input.size(),
                               masTempBuffer.d_pointer(),
                               masTempBuffer.sizeInBytes,

                               masOutputBuffer.d_pointer(),
                               masOutputBuffer.sizeInBytes,

                               &mas,

                               &emitDesc,1
                               ));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // perform compaction
   // ==================================================================
   uint64_t compactedSize;
   compactedSizeBuffer.download(&compactedSize,1);

   masBuffer.alloc(compactedSize);
   OPTIX_CHECK(optixAccelCompact(optixContext,
                                 0, //stream
                                 mas,
                                 masBuffer.d_pointer(),
                                 masBuffer.sizeInBytes,
                                 &mas));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // aaaaaand .... clean up
   // ==================================================================
   //compactedSizeBuffer.free();
}

void OptixRender::buildVolumeAccel(){
    const int nb_obj_volumic = modele->getNumVolume();
    OptixBuildInput *input = (OptixBuildInput*)malloc(sizeof(OptixBuildInput) *nb_obj_volumic );
    
    //Create accel for volume
    aabb = (OptixAabb*)malloc(sizeof(OptixAabb)*nb_obj_volumic);

      for(size_t i = 0; i < nb_obj_volumic; ++i){
          modele->getAabb(i,reinterpret_cast<float*>(&aabb[i]));
      }
    
    cudaMalloc(reinterpret_cast<void**>(&d_aabb
        ),nb_obj_volumic*sizeof(OptixAabb));
    cudaMemcpy(
        reinterpret_cast<void*>(d_aabb),
        aabb,
        nb_obj_volumic*sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    );
   uint32_t aabb_input_flags[] = {
       OPTIX_GEOMETRY_FLAG_NONE
   };

   OptixBuildInput aabb_input = {};
   aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
   aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb;
   aabb_input.customPrimitiveArray.flags = aabb_input_flags;
   aabb_input.customPrimitiveArray.numSbtRecords = 1;
   aabb_input.customPrimitiveArray.numPrimitives = nb_obj_volumic;
   aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
   aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(sbtData);
   aabb_input.customPrimitiveArray.strideInBytes = 0;
   aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

   for(int i = 0; i < nb_obj_volumic; ++i)
        input[i] = aabb_input;

   // ==================================================================
   // BLAS setup
   // ==================================================================


   OptixAccelBuildOptions accelOptions;

       accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE
         | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
         ;
       accelOptions.motionOptions.numKeys  = 1;
       accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;


   OptixAccelBufferSizes blasBufferSizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage
               (optixContext,
                &accelOptions,
                input,
                nb_obj_volumic,  // num_build_inputs
                &blasBufferSizes
                ));

   // ==================================================================
   // prepare compaction
   // ==================================================================

   CUDABuffer compactedSizeBuffer;
   compactedSizeBuffer.alloc(sizeof(uint64_t));

   OptixAccelEmitDesc emitDesc;
   emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emitDesc.result = compactedSizeBuffer.d_pointer();

   // ==================================================================
   // execute build (main stage)
   // ==================================================================

   vasTempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
   vasOutputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

   OPTIX_CHECK(optixAccelBuild(optixContext,
                               0, //stream
                               &accelOptions,
                               input,
                               nb_obj_volumic,
                               vasTempBuffer.d_pointer(),
                               vasTempBuffer.sizeInBytes,

                               vasOutputBuffer.d_pointer(),
                               vasOutputBuffer.sizeInBytes,

                               &vas,

                               &emitDesc,1
                               ));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // perform compaction
   // ==================================================================
   uint64_t compactedSize;
   compactedSizeBuffer.download(&compactedSize,1);

   vasBuffer.alloc(compactedSize);
   OPTIX_CHECK(optixAccelCompact(optixContext,
                                 0,//stream,
                                 vas,
                                 vasBuffer.d_pointer(),
                                 vasBuffer.sizeInBytes,
                                 &vas));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // aaaaaand .... clean up
   // ==================================================================
   compactedSizeBuffer.free();
   free(input);
}

void OptixRender::updateSBT(){
    CUDA_SYNC_CHECK();
    for(size_t i = 0; i < modele->getNumVolume(); ++i){
        HitgroupRecord& rec = hitgroupRecords[i];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[VOLUME_RENDERING],&rec));
        modele->getVolumeSbt(i,&rec.sbt);

    }
    //Create HitGroup for mesh object*
    const int numMeshObjects = (int)modele->getNumMesh();
    for (int i=0; i<numMeshObjects;i++) {
      HitgroupRecord& rec =  hitgroupRecords[i + modele->getNumVolume()];
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[MESH_RENDERING],&rec));
      modele->getMeshSbt(i,&rec.sbt);
    }
}

void OptixRender::updateAccelerationStructure(){
    CUDA_SYNC_CHECK();

    if(this->isVolumeDataModified){
        vas = {0};
        if(this->modele->getNumVolume() > 0) updateVAS();
    }
    if(this->isMeshDataModified) {
        mas = {0};
        if(this->modele->getNumMesh() > 0) updateMAS();
    }

    if(isVolumeDataModified | isMeshDataModified)
    {
        updateIAS();
        launchParams.traversable = ias;
    }
    if(isSBTDataModified){
         updateSBT();
         hitgroupRecordsBuffer.free();
         hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
         sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
         sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
         sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
    }
    isVolumeDataModified = false;
    isMeshDataModified = false;
    isSBTDataModified = false;

}
//Mets à jours les données concernants les Meshs
void OptixRender::updateMAS(){
    CUDA_SYNC_CHECK();
    size_t nb_obj_mesh = modele->getNumMesh();
  // ==================================================================
  // triangle inputs
  // ==================================================================
  // upload the model to the device: the builder
  std::vector<OptixBuildInput> input;
  input.resize(nb_obj_mesh);
  std::vector<CUdeviceptr> d_vertices ;
  std::vector<CUdeviceptr> d_indices;
  d_vertices.resize(nb_obj_mesh);
  d_indices.resize(nb_obj_mesh);
  uint32_t triangleInputFlags[1] = { 0 };
  for(int i = 0; i < (int)nb_obj_mesh; ++i){

    input[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[i] = modele->getMesh(i)->getVertexDevicePointer();
    d_indices[i]  = modele->getMesh(i)->getIndexDevicePointer();
    input[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input[i].triangleArray.vertexStrideInBytes = sizeof(vec3f);
    input[i].triangleArray.numVertices = (int)modele->getMesh(i)->getNumVertices();
    input[i].triangleArray.vertexBuffers = &d_vertices[i];

    input[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input[i].triangleArray.indexStrideInBytes = sizeof(vec3i);
    input[i].triangleArray.numIndexTriplets = (int)modele->getMesh(i)->getNumIndex();
    input[i].triangleArray.indexBuffer = d_indices[i];

    // in this example we have one SBT entry, and no per-primitive
              // materials:
    input[i].triangleArray.flags = triangleInputFlags;
    input[i].triangleArray.numSbtRecords = 1;
    input[i].triangleArray.sbtIndexOffsetBuffer = 0;
    input[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
    input[i].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(sbtData);

  }
   // ==================================================================
   // BLAS setup
   // ==================================================================
   OptixAccelBuildOptions accelOptions;

       accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE
         | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
         ;
       accelOptions.motionOptions.numKeys  = 1;
       accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;


   OptixAccelBufferSizes blasBufferSizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage
               (optixContext,
                &accelOptions,
                input.data(),
                (int)input.size(),  // num_build_inputs
                &blasBufferSizes
                ));

   // ==================================================================
   // prepare compaction
   // ==================================================================

   CUDABuffer compactedSizeBuffer;
   compactedSizeBuffer.alloc(sizeof(uint64_t));

   OptixAccelEmitDesc emitDesc;
   emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emitDesc.result = compactedSizeBuffer.d_pointer();

   // ==================================================================
   // execute build (main stage)
   // ==================================================================

   if(masTempBuffer.sizeInBytes != blasBufferSizes.tempSizeInBytes)
     masTempBuffer.resize(blasBufferSizes.tempSizeInBytes);

   if(masOutputBuffer.sizeInBytes != blasBufferSizes.outputSizeInBytes)
        masOutputBuffer.resize(blasBufferSizes.outputSizeInBytes);

   OPTIX_CHECK(optixAccelBuild(optixContext,
                               /* stream */0,
                               &accelOptions,
                               input.data(),
                               (int)input.size(),
                               masTempBuffer.d_pointer(),
                               masTempBuffer.sizeInBytes,

                               masOutputBuffer.d_pointer(),
                               masOutputBuffer.sizeInBytes,

                               &mas,

                               &emitDesc,1
                               ));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // perform compaction
   // ==================================================================
   uint64_t compactedSize;
   compactedSizeBuffer.download(&compactedSize,1);

   masBuffer.free();
   masBuffer.alloc(compactedSize);
   OPTIX_CHECK(optixAccelCompact(optixContext,
                                 /*stream:*/0,
                                 mas,
                                 masBuffer.d_pointer(),
                                 masBuffer.sizeInBytes,
                                 &mas));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // aaaaaand .... clean up
   // ==================================================================
   compactedSizeBuffer.free();
}
//Mets à jours les données concernants les aabb structures
void OptixRender::updateVAS(){
    const int nb_obj_volumic = modele->getNumVolume();
    OptixBuildInput *input = (OptixBuildInput*)malloc(sizeof(OptixBuildInput) *nb_obj_volumic );

    //Create accel for volume
    free(aabb);
    aabb = (OptixAabb*)malloc(sizeof(OptixAabb)*nb_obj_volumic);

    for(size_t i = 0; i < nb_obj_volumic; ++i){
        modele->getAabb(i,reinterpret_cast<float*>(&aabb[i]));
    }

    cudaMalloc(reinterpret_cast<void**>(&d_aabb
        ),nb_obj_volumic*sizeof(OptixAabb));

    cudaMemcpy(
        reinterpret_cast<void*>(d_aabb),
        aabb,
        nb_obj_volumic*sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    );
    uint32_t aabb_input_flags[] = {
       OPTIX_GEOMETRY_FLAG_NONE
    };
   OptixBuildInput aabb_input = {};
   aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
   aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb;
   aabb_input.customPrimitiveArray.flags = aabb_input_flags;
   aabb_input.customPrimitiveArray.numSbtRecords = 1;
   aabb_input.customPrimitiveArray.numPrimitives = nb_obj_volumic;
   aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
   aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(sbtData);
   aabb_input.customPrimitiveArray.strideInBytes = 0;
   aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

   for(int i = 0; i < nb_obj_volumic; ++i)
        input[i] = aabb_input;

   // ==================================================================
   // BLAS setup
   // ==================================================================


   OptixAccelBuildOptions accelOptions;

       accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE
         | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
         ;
       accelOptions.motionOptions.numKeys  = 1;
       accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;


   OptixAccelBufferSizes blasBufferSizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage
               (optixContext,
                &accelOptions,
                input,
                nb_obj_volumic,  // num_build_inputs
                &blasBufferSizes
                ));

   // ==================================================================
   // prepare compaction
   // ==================================================================

   CUDABuffer compactedSizeBuffer;
   compactedSizeBuffer.alloc(sizeof(uint64_t));

   OptixAccelEmitDesc emitDesc;
   emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emitDesc.result = compactedSizeBuffer.d_pointer();

   // ==================================================================
   // execute build (main stage)
   // ==================================================================

   vasTempBuffer.free();
   vasTempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

   vasOutputBuffer.free();
   vasOutputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

   OPTIX_CHECK(optixAccelBuild(optixContext,
                               /* stream */0,
                               &accelOptions,
                               input,
                               nb_obj_volumic,
                               vasTempBuffer.d_pointer(),
                               vasTempBuffer.sizeInBytes,

                               vasOutputBuffer.d_pointer(),
                               vasOutputBuffer.sizeInBytes,

                               &vas,

                               &emitDesc,1
                               ));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // perform compaction
   // ==================================================================
   uint64_t compactedSize;
   compactedSizeBuffer.download(&compactedSize,1);

    vasBuffer.free();
   vasBuffer.alloc(compactedSize);
   OPTIX_CHECK(optixAccelCompact(optixContext,
                                 /*stream:*/0,
                                 vas,
                                 vasBuffer.d_pointer(),
                                 vasBuffer.sizeInBytes,
                                 &vas));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // aaaaaand .... clean up
   // ==================================================================
   compactedSizeBuffer.free();
   free(input);
}

void OptixRender::updateIAS(){
    CUDA_SYNC_CHECK();
    std::vector<OptixInstance> instances;
    instances.resize(2);

    float transform[12];
    transform[0] = 1.0f; transform[1] = 0.0f; transform[2] = 0.0f; transform[3] = 0.0f;
    transform[4] = 0.0f; transform[5] = 1.0f; transform[6] = 0.0f; transform[7] = 0.0f;
    transform[8] = 0.0f; transform[9] = 0.0f; transform[10] = 1.0f; transform[11] = 0.0f;
    //instances 0 for volume
    size_t sbtOffset = 0;
    memcpy(instances[0].transform, transform, sizeof(float)*12);
    instances[0].instanceId = 0;
    instances[0].sbtOffset = (int)sbtOffset;
    instances[0].visibilityMask = 0xFF;
    instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
    instances[0].traversableHandle = vas;
    sbtOffset +=  modele->getNumVolume();

    //instance 1 for mesh

    memcpy(instances[1].transform, transform, sizeof(float)*12);
    instances[1].instanceId = 0;
    instances[1].sbtOffset = (int)sbtOffset;
    instances[1].visibilityMask = 0xFF;
    instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
    instances[1].traversableHandle = mas;
    sbtOffset += modele->getNumMesh();

    CUDABuffer deviceInstances;
    deviceInstances.alloc_and_upload(instances);

    OptixBuildInput input;
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = deviceInstances.d_pointer();
    input.instanceArray.numInstances = (int)instances.size();
   // ==================================================================
   // BLAS setup
   // ==================================================================

   OptixAccelBuildOptions accelOptions;

   accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE
     | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
     ;
   accelOptions.motionOptions.numKeys  = 1;
   accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;


   OptixAccelBufferSizes blasBufferSizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage
               (optixContext,
                &accelOptions,
                &input,
                1,  // num_build_inputs
                &blasBufferSizes
                ));

   // ==================================================================
   // prepare compaction
   // ==================================================================

   CUDABuffer compactedSizeBuffer;
   compactedSizeBuffer.alloc(sizeof(uint64_t));

   OptixAccelEmitDesc emitDesc;
   emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emitDesc.result = compactedSizeBuffer.d_pointer();

   // ==================================================================
   // execute build (main stage)
   // ==================================================================

   OPTIX_CHECK(optixAccelBuild(optixContext,
                               0, //stream
                               &accelOptions,
                               &input,
                               1,
                               iasTempBuffer.d_pointer(),
                               iasTempBuffer.sizeInBytes,

                               iasOutputBuffer.d_pointer(),
                               iasOutputBuffer.sizeInBytes,

                               &ias,

                               &emitDesc,1
                               ));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // perform compaction
   // ==================================================================
   uint64_t compactedSize;
   compactedSizeBuffer.download(&compactedSize,1);

   iasBuffer.free();
   iasBuffer.alloc(compactedSize);
   OPTIX_CHECK(optixAccelCompact(optixContext,
                                 0, //stream
                                 ias,
                                 iasBuffer.d_pointer(),
                                 iasBuffer.sizeInBytes,
                                 &ias));
   CUDA_SYNC_CHECK();

   // ==================================================================
   // aaaaaand .... clean up
   // ==================================================================-
   compactedSizeBuffer.free();

}


void OptixRender::render(){
    CUDA_SYNC_CHECK();
    if (launchParams.frame.size.x == 0) return;
     CUDA_SYNC_CHECK();
    updateAccelerationStructure();
     CUDA_SYNC_CHECK();
    launchParamsBuffer.upload(&launchParams,1,stream);

     CUDA_SYNC_CHECK();
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                                pipeline,stream,
                                /*! parameters and SBT */
                                launchParamsBuffer.d_pointer(),
                                launchParamsBuffer.sizeInBytes,
                                &sbt,
                                /*! dimensions of the launch: */
                                launchParams.frame.size.x,
                                launchParams.frame.size.y,
                                1
                                ));
    CUDA_SYNC_CHECK();
}
void OptixRender::resize(const vec2i &newSize)
{
    CUDA_SYNC_CHECK();
      // if window minimized
      if ((newSize.x == 0) || (newSize.y == 0)) return;
      // resize our cuda frame buffer
      colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));

      // update the launch parameters that we'll pass to the optix
      // launch:
      launchParams.frame.size = newSize;

      launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_ptr;
}

/*! download the rendered color buffer */
void OptixRender::downloadPixels(uint32_t h_pixels[]){
    CUDA_SYNC_CHECK();
    colorBuffer.download(h_pixels,0,0,
                             launchParams.frame.size.x*launchParams.frame.size.y, stream);
}


void OptixRender::setCamera(const Camera &cam){
    launchParams.camera.position  = cam.pos;
    launchParams.camera.direction = normalize(cam.at - cam.pos);
    launchParams.camera.up = cam.up;

    const float cosFovy = 0.66f;

    const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
    launchParams.camera.horizontal
      = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                                           cam.up));
    launchParams.camera.vertical
      = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
}
/*! set camera to render with */
void OptixRender::updateSBTBuffer(){
    CUDA_SYNC_CHECK();
    hitgroupRecords.resize(modele->getNumVolume() + modele->getNumMesh());
}

void OptixRender::notifyMeshChanges(){
    this->isMeshDataModified = true;
}
void OptixRender::notifySbtChanges(){
    isSBTDataModified = true;
}
void OptixRender::notifyAabbObjectChanges(){
    isVolumeDataModified = true;
}
