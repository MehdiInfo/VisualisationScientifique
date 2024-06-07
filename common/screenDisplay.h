#ifndef __SCREENDISPLAY_H__
#define __SCREENDISPLAY_H__

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>
#include <vector>

#include "vec.h"
#include "camera.h"
#include "Scene.hpp"
#include "optixRender.h"
#include "mmatrix.h"
#include "meshloader.h"

class ScreenDisplay {
    public :

        ScreenDisplay(const int width = 680, const int height = 420,const std::string title = "VSProject");
        ~ScreenDisplay();

        void createSceneEntities();

        void run();
        
        void updateInterface();
        void update();
        void render();
        void drawScene();

        void resize(const int width, const int height);

        vec2i getSize() const;
        int getWidth()  const;
        int getHeight() const;

    private :
    GLFWwindow* window;
    std::string m_windowTitle = "VSProject" ;
    std::vector<uint32_t> pixels;
    GLuint                fbTexture {0};
    Scene scene;
    OptixRender *optixRender;

public : 

static bool translation;
static bool rotation;
static vec3f translateCamera;
static vec2f oldCursorPosition;
static Camera m_camera;
static vec2f coordonneeSpherique;
static vec2f ihmpos;
static vec2f ihmsize;
static vec2i m_screenSize;
static bool updated;
};


#endif