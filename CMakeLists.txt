cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mdstress_test LANGUAGES CXX CUDA)

# set fast math etc
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-deprecated-gpu-targets -arch=sm_52" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# find lapack library
FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)

# set our cmake modules path for tclap
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# find all the source files
file(GLOB SOURCES src/*.cpp src/*.cu contrib/voro++/src/*.cc)

# remove sources
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/voro++.cc)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/v_base_wl.cc)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/cmd_line.cc)

# include the headers directory
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include/mdstress include/voro++)
link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/lib64)

add_executable(mdstress_test ${SOURCES})
set_target_properties(mdstress_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(mdstress_test PRIVATE cuda ${LAPACK_LIBRARY})

INSTALL(TARGETS mdstress_test RUNTIME DESTINATION bin)
