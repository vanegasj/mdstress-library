cmake_minimum_required(VERSION 2.8)
find_package(PythonLibs 3.6 REQUIRED )
find_package(PythonInterp 3.6 REQUIRED )

enable_language(C)
enable_language(CXX)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

option(MDS_BOOSTPYTHON "Enable Boost Python bindings" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)

FILE(GLOB MDSTRESS_SOURCES_H   include/mdstress/*.h)
FILE(GLOB MDSTRESS_SOURCES_CPP src/*.cpp)

# the voro++ build process: treats voro++ as header-only library
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/contrib/voro++-0.4.6)
    SET(VORO_URL http://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz)
    SET(VORO_DOWNLOAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++-0.4.6.tar.gz)
    message("Downloading and extracting Voro++")
    FILE(DOWNLOAD ${VORO_URL} ${VORO_DOWNLOAD_PATH})
    EXECUTE_PROCESS(
        COMMAND tar xaf voro++-0.4.6.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
endif()
FILE(GLOB VORO_SOURCES_HH contrib/voro++-0.4.6/src/*.hh)
FILE(GLOB VORO_SOURCES_CC contrib/voro++-0.4.6/src/*.cc)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/voro++)
FILE(COPY ${VORO_SOURCES_HH} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include/voro++)
LIST(REMOVE_ITEM VORO_SOURCES_CC ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++-0.4.6/src/voro++.cc)
LIST(REMOVE_ITEM VORO_SOURCES_CC ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++-0.4.6/src/v_base_wl.cc)

SET(MDSTRESS_SOURCES ${MDSTRESS_SOURCES_CPP} ${MDSTRESS_SOURCES_H} ${VORO_SOURCES_CC} ${VORO_SOURCES_HH})

INCLUDE_DIRECTORIES(include/mdstress)
INCLUDE_DIRECTORIES(contrib/voro++-0.4.6/src)

# attempt to find LAPACK library
FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)
if (LAPACK_LIBRARY-NOTFOUND)
    # check to see if we need to download LAPACK
    SET(LAPACK_URL http://www.netlib.org/lapack/lapack-3.8.0.tar.gz)
    SET(LAPACK_DOWNLOAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib/lapack-3.8.0.tar.gz)
    FILE(DOWNLOAD ${LAPACK_URL} ${LAPACK_DOWNLOAD_PATH})
    EXECUTE_PROCESS(
        COMMAND tar xaf lapack-3.8.0.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
    ADD_SUBDIRECTORY(contrib/lapack-3.8.0)
endif()

# detect if we have a parent and compile statically
GET_DIRECTORY_PROPERTY(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    ADD_LIBRARY(mdstress STATIC ${MDSTRESS_SOURCES})
else()
    ADD_LIBRARY(mdstress SHARED ${MDSTRESS_SOURCES})
endif()

TARGET_LINK_LIBRARIES(mdstress ${LAPACK_LIBRARY} ${PYTHON_LIBRARIES})

INSTALL(TARGETS mdstress RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib/static)

INSTALL(FILES ${MDSTRESS_SOURCES_H} DESTINATION include/mdstress)
INSTALL(FILES ${VORO_SOURCES_HH} DESTINATION include/voro++)

# only process python bindings if requested
if(MDS_BOOSTPYTHON)
    ADD_SUBDIRECTORY(boostpython)
endif()

# process tensortools
ADD_SUBDIRECTORY(tensortools)
