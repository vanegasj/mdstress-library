cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(mdstress_test LANGUAGES CXX )
#cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
#project(mdstress_test LANGUAGES CXX CUDA)

# set the C++11 standard
set(CMAKE_CXX_STANDARD 11)

# set fast math etc
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)
#set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-deprecated-gpu-targets -use_fast_math -restrict -Xptxas -O3,-v -Xcompiler -O3 -arch=sm_61" CACHE STRING
#    "Flags used by the compiler during release builds." FORCE)

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set the boostpython option
option(MDS_BOOSTPYTHON "Enable Boost Python bindings" OFF)

set(EXT_PRECISION "EXT_DOUBLE" CACHE STRING "Precision used by exposed function calls")
set_property(CACHE EXT_PRECISION PROPERTY STRINGS EXT_DOUBLE EXT_FLOAT)
set(INT_PRECISION "INT_DOUBLE" CACHE STRING "Precision used internally")
set_property(CACHE INT_PRECISION PROPERTY STRINGS INT_FLOAT INT_DOUBLE INT_LONG_DOUBLE)

# find lapack library
find_library(LAPACK_LIBRARY_SYS NAMES lapack liblapack HINTS /usr/lib REQUIRED)
if (NOT LAPACK_LIBRARY_SYS)
    unset(LAPACK_LIBRARY_SYS-NOTFOUND)
    unset(LAPACK_LIBRARY_SYS)
    # check to see if we need to download LAPACK
    set(LAPACK_URL http://www.netlib.org/lapack/lapack-3.8.0.tar.gz)
    set(LAPACK_DOWNLOAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib/lapack-3.8.0.tar.gz)
    file(DOWNLOAD ${LAPACK_URL} ${LAPACK_DOWNLOAD_PATH})
    execute_process(
        COMMAND tar xaf lapack-3.8.0.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
    add_subdirectory(contrib/lapack-3.8.0)
else()
    find_library(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)
endif()

# set our cmake modules path for tclap
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# create our mds_defines.h file in the binary build directory
configure_file(include/mdstress/mds_defines.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/mdstress/mds_defines.h @ONLY)

# find all the source files
file(GLOB SOURCES src/*.cpp src/*.cu contrib/voro++/src/*.cc)

# the include files
file(GLOB MDSTRESS_SOURCES_H include/mdstress/*.h )
file(GLOB VORO_SOURCES_HH include/voro++/*.hh)

# remove sources
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/voro++.cc)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/v_base_wl.cc)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/cmd_line.cc)

# include the headers directory
include_directories(include/mdstress include/voro++ ${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include/mdstress include/voro++)
#link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/lib64)

add_library(mdstress SHARED ${SOURCES})
#set_target_properties(mdstress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(mdstress PRIVATE ${LAPACK_LIBRARY})

install(TARGETS mdstress RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib/static)
install(FILES ${MDSTRESS_SOURCES_H} DESTINATION include/mdstress)
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mds_defines.h DESTINATION include/mdstress)
install(FILES ${VORO_SOURCES_HH} DESTINATION include/voro++)

# only process python bindings if requested
if(MDS_BOOSTPYTHON)
    add_subdirectory(boostpython)
endif()

# process tensortools
add_subdirectory(tensortools)

if(MODULEFILES_PATH)
    # modify the file and copy it to build dir
    configure_file(mdstress-library.in mdstress-library @ONLY)

    # install it in the specified modulesfiles dir
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mdstress-library DESTINATION ${MODULEFILES_PATH} PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
endif(MODULEFILES_PATH)
