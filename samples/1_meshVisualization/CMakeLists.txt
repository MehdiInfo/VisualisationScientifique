add_executable(1_meshVisualization
  ./main.cpp
  ${CMAKE_SOURCE_DIR}/common/screenDisplay.cpp
  ${CMAKE_SOURCE_DIR}/common/Scene.cpp
  ${CMAKE_SOURCE_DIR}/common/image.cpp
  ${CMAKE_SOURCE_DIR}/common/trianglemesh.cpp
  ./optixRender.h
  ./optixRender.cpp
  ${CMAKE_SOURCE_DIR}/common/meshloader.cpp
  ${CMAKE_SOURCE_DIR}/common/volume.cpp
  ${CMAKE_SOURCE_DIR}/common/mmatrix.cpp
)

target_link_libraries(1_meshVisualization
  OpenGL::GL
  imgui
  glfw
  assimp
  ${OPENGL_LIBRARY}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )

