cmake_minimum_required(VERSION 3.9)
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
project(VSProject LANGUAGES CXX VERSION 0.0.0)

if (NOT WIN32)
# visual studio doesn't like these (not need them):
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE LEGACY)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE) #Desactive la génération de la doc
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE) #Desactive les tests
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) #Desactive la compilation des exemples
find_package(OpenGL REQUIRED)

set(glfw_dir ${PROJECT_SOURCE_DIR}/libs/glfw3/)
include_directories(${glfw_dir}/include)
add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# build glew
# ------------------------------------------------------------------
set(glew_dir ${PROJECT_SOURCE_DIR}/libs/glew-2.1.0/)
include_directories(./libs/glew-2.1.0/include)
add_subdirectory(./libs/glew-2.1.0/build/cmake/) #chemin vers les sources de GLew 2.1.0

# ------------------------------------------------------------------
# build Assimp
# ------------------------------------------------------------------
# set variables that are needed
include_directories(./libs/assimp/include)
link_directories(./libs/assimp/lib64)
link_directories(./libs/assimp/bin)

# ------------------------------------------------------------------
# build sbt
# ------------------------------------------------------------------
include_directories(./libs/stb/)


# ------------------------------------------------------------------
# build OptiX
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

include_directories(${OptiX_INCLUDE})
include_directories(${CMAKE_SOURCE_DIR}/common/)

# ImGUI
set(IMGUI_DIR "./libs/imgui/")

add_library("imgui" "${IMGUI_DIR}/imgui.cpp"
                  "${IMGUI_DIR}/imgui_demo.cpp"
                  "${IMGUI_DIR}/imgui_draw.cpp"
                  "${IMGUI_DIR}/imgui_tables.cpp"
                  "${IMGUI_DIR}/imgui_widgets.cpp"
                  
                  "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
                  "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp")

include_directories("${IMGUI_DIR}")
#target_include_directories(VSProject PRIVATE "${IMGUI_DIR}")
include_directories("${IMGUI_DIR}/backends/")
#target_include_directories(VSProject PRIVATE "${IMGUI_DIR}/backends/")
#target_link_libraries(VSProject "imgui" "${CMAKE_DL_LIBS}")

# ------------------------------------------------------------------
# build Samples
# ------------------------------------------------------------------
add_subdirectory(samples)

add_executable(VSProject
  ${embedded_ptx_code}
  ./common/main.cpp
  ./common/screenDisplay.cpp
  ./common/Scene.cpp
  ./common/trianglemesh.cpp
  ./common/optixRender.cpp
  ./common/volume.cpp
  ./common/mmatrix.cpp
  ./common/image.cpp
  ./common/meshloader.cpp
)


include_directories(${OptiX_INCLUDE})

add_definitions(
    -DTW_STATIC
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
)

target_link_libraries(VSProject
  OpenGL::GL
  imgui
  glfw
  assimp
  ${OPENGL_LIBRARY}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )


#compile shader
enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

# CUDA PACKAGE
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER clang++)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


# COMPILE CU FILES
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(mesh OBJECT ${CMAKE_SOURCE_DIR}/shader/mesh.cu)
set_property(TARGET mesh PROPERTY CUDA_ARCHITECTURES 50-virtual)
set_target_properties(mesh PROPERTIES
CUDA_PTX_COMPILATION ON)

add_library(raygen OBJECT ${CMAKE_SOURCE_DIR}/shader/raygen.cu)
set_property(TARGET raygen PROPERTY CUDA_ARCHITECTURES 50-virtual)
set_target_properties(raygen PROPERTIES
CUDA_PTX_COMPILATION ON)

add_library(volume OBJECT ${CMAKE_SOURCE_DIR}/shader/volume.cu)
set_property(TARGET volume PROPERTY CUDA_ARCHITECTURES 50-virtual)
set_target_properties(volume PROPERTIES CUDA_PTX_COMPILATION ON)

#add_custom_command(TARGET VSProject POST_BUILD  COMMAND ${CMAKE_COMMAND} -E copy
#                       ${CMAKE_BINARY_DIR}/CMakeFiles/volume.dir/shader/volume.ptx ${CMAKE_BINARY_DIR}/volume.ptx)

#add_custom_command(TARGET VSProject POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#                       ${CMAKE_BINARY_DIR}/raygen.dir/$<CONFIG>/raygen.ptx $<TARGET_FILE_DIR:VSProject>/raygen.ptx)

#add_custom_command(TARGET VSProject POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#                       ${CMAKE_BINARY_DIR}/mesh.dir/$<CONFIG>/mesh.ptx $<TARGET_FILE_DIR:VSProject>/mesh.ptx)

#add_custom_command(TARGET VSProject POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#                       ${CMAKE_SOURCE_DIR}/libs/assimp/bin/assimp-vc142-mt.dll $<TARGET_FILE_DIR:VSProject>/assimp-vc142-mt.dll)


