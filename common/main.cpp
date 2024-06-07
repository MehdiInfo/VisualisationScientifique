#define MIXED_VISUALIZATION 1
#define TEST 1
#include <cstdlib>
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "screenDisplay.h"
#define STB_IMAGE_IMPLEMENTATION
#include "image.h"


#define WIDTH 1920
#define HEIGHT 1280

bool ScreenDisplay::translation;
bool ScreenDisplay::rotation;
bool ScreenDisplay::updated;
vec3f ScreenDisplay::translateCamera;
vec2f ScreenDisplay::oldCursorPosition;
Camera ScreenDisplay::m_camera;
vec2f ScreenDisplay::coordonneeSpherique;
vec2f ScreenDisplay::ihmpos; 
vec2f ScreenDisplay::ihmsize;
vec2i ScreenDisplay::m_screenSize;


void compute_one_frame();

int main(int argc, char** argv){

    if (argc == 2 ){
        if (strcmp(argv[1],"--offscreen") == 0  ){
            compute_one_frame();
        }
        else {
            std::cout << "Program take juste one argument : --offscreen to compute just one image in background" << std::endl;            
        }
        return EXIT_SUCCESS;
    }
    ScreenDisplay w(680,420);
    w.run();
    return EXIT_SUCCESS;
}

void compute_one_frame(){

    Camera m_camera; //default look at camera
    std::vector<uint32_t> pixels; //framebuffer
    pixels.resize(WIDTH*HEIGHT); //standard dimension
    Scene scene; //Scene description
    OptixRender *optixRender; //OptiX Context
    
    //Initialize scene
    MeshLoader loader;
    loader.LoadMesh("../data/statue/statue.obj");
    std::vector<TriangleMesh*> meshs;
    meshs = loader.toTriangleMeshs();
    for(size_t i = 0; i < meshs.size(); ++i)
        scene.addMesh(meshs[i]);
    Volume *v = new Volume();
    v->loadVolume("../data/cafard.dat");
    v->translate(vec3f(1.f,0.f,0.f));
    scene.addVolume(v);

    //Initialize OptiX Context
    optixRender = new OptixRender(&scene);
    optixRender->resize(vec2i(WIDTH,HEIGHT));

    optixRender->setCamera(m_camera);
    //Make rendering

    optixRender->render();
    cudaDeviceSynchronize();
    optixRender->downloadPixels(pixels.data());
    cudaDeviceSynchronize();
    std::cout << "size : " << pixels.size() << std::endl;
    saveBMP("./result.bmp", WIDTH,HEIGHT, pixels.data());  
}