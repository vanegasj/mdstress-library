cmake_minimum_required(VERSION 2.8)
find_package(PythonLibs 2.7 REQUIRED )
find_package(PythonInterp 2.7 REQUIRED )

enable_language(C)
enable_language(CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)

FILE(GLOB MDSTRESS_SOURCES_H   include/*.h)
FILE(GLOB MDSTRESS_SOURCES_CPP src/*.cpp)

# the voro++ sources, sans voro++.cc
FILE(GLOB VORO_SOURCES_HH   contrib/voro++/*.hh)
FILE(GLOB VORO_SOURCES_CC contrib/voro++/*.cc)
SET(MDSTRESS_SOURCES ${MDSTRESS_SOURCES_CPP} ${MDSTRESS_SOURCES_H} ${VORO_SOURCES_CC} ${VORO_SOURCES_HH})


INCLUDE_DIRECTORIES(include/)
INCLUDE_DIRECTORIES(contrib/voro++/)

FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)

FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)

ADD_LIBRARY(mdstress SHARED ${MDSTRESS_SOURCES})

TARGET_LINK_LIBRARIES(mdstress ${LAPACK_LIBRARY} ${PYTHON_LIBRARIES})

INSTALL(TARGETS mdstress RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib/static)

INSTALL(FILES ${MDSTRESS_SOURCES_H} ${VORO_SOURCES_HH} DESTINATION include/mdstress)

ADD_SUBDIRECTORY(boostpython)
ADD_SUBDIRECTORY(tensortools)
