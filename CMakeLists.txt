cmake_minimum_required(VERSION 2.8)
find_package(PythonLibs 3.6 REQUIRED )
find_package(PythonInterp 3.6 REQUIRED )

enable_language(C)
enable_language(CXX)

option(MDS_BOOSTPYTHON "Enable Boost Python bindings" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)

FILE(GLOB MDSTRESS_SOURCES_H   include/*.h)
FILE(GLOB MDSTRESS_SOURCES_CPP src/*.cpp)

# the voro++ build process
SET(VORO_URL http://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz)
SET(VORO_DOWNLOAD_PATH ${PROJECT_SOURCE_DIR}/contrib/voro++-0.4.6.tar.gz)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/contrib/voro++-0.4.6)
    FILE(DOWNLOAD ${VORO_URL} ${VORO_DOWNLOAD_PATH})
    EXECUTE_PROCESS(
        COMMAND tar xaf voro++-0.4.6.tar.gz
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/contrib)
endif()
FILE(GLOB VORO_SOURCES_HH contrib/voro++-0.4.6/src/*.hh)
FILE(GLOB VORO_SOURCES_CC contrib/voro++-0.4.6/src/*.cc)
LIST(REMOVE_ITEM VORO_SOURCES_CC ${PROJECT_SOURCE_DIR}/contrib/voro++-0.4.6/src/voro++.cc)
LIST(REMOVE_ITEM VORO_SOURCES_CC ${PROJECT_SOURCE_DIR}/contrib/voro++-0.4.6/src/v_base_wl.cc)

SET(MDSTRESS_SOURCES ${MDSTRESS_SOURCES_CPP} ${MDSTRESS_SOURCES_H} ${VORO_SOURCES_CC} ${VORO_SOURCES_HH})

INCLUDE_DIRECTORIES(include/)
INCLUDE_DIRECTORIES(contrib/voro++-0.4.6/src)

FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)
ADD_LIBRARY(mdstress SHARED ${MDSTRESS_SOURCES})

TARGET_LINK_LIBRARIES(mdstress ${LAPACK_LIBRARY} ${PYTHON_LIBRARIES})

INSTALL(TARGETS mdstress RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib/static)

INSTALL(FILES ${MDSTRESS_SOURCES_H} ${VORO_SOURCES_HH} DESTINATION include/mdstress)

# only process python bindings if requested
if(MDS_BOOSTPYTHON)
    ADD_SUBDIRECTORY(boostpython)
endif()

# process tensortools
ADD_SUBDIRECTORY(tensortools)
