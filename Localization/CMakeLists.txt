cmake_minimum_required(VERSION 2.6)

project(alignTwoRGBDGPU)
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
find_package(GLUT REQUIRED)



if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -std=c++0x ")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;--compiler-options;-O2;-DVERBOSE") 
  endif()
endif()

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
  estimatRTGPU.cu # gpu ransac
  dataTrain.cu
)  

set(sources
  #estimatRT.cpp # cpu ransac
  mainSift.cpp
  renderMesh.cpp
  RGBD_utils.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${GLUT_INCLUDE_DIRS}
)
include_directories (/opt/X11/include/)

cuda_add_executable(alignTwoRGBDGPU ${cuda_sources} ${sources})


set_target_properties(alignTwoRGBDGPU PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(alignTwoRGBDGPU
  "${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}"
  /opt/X11/lib/libOSMesa.dylib
)


install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)