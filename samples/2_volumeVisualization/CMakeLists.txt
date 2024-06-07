add_executable(2_volumeVisualization
  ./main.cpp
  ${CMAKE_SOURCE_DIR}/common/screenDisplay.cpp
  ${CMAKE_SOURCE_DIR}/common/Scene.cpp
  ${CMAKE_SOURCE_DIR}/common/trianglemesh.cpp
  ./optixRender.cpp
  ${CMAKE_SOURCE_DIR}/common/volume.cpp
  ${CMAKE_SOURCE_DIR}/common/image.cpp
  ${CMAKE_SOURCE_DIR}/common/meshloader.cpp
  ${CMAKE_SOURCE_DIR}/common/mmatrix.cpp
  
)

target_link_libraries(2_volumeVisualization
  OpenGL::GL
  imgui
  glfw
  assimp
  ${OPENGL_LIBRARY}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
