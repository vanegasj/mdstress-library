cmake_minimum_required(VERSION 2.8)

enable_language(C)
enable_language(CXX)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

option(MDS_BOOSTPYTHON "Enable Boost Python bindings" OFF)
option(MDS_COMPILEMAIN "Compile main.cpp into a target" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)

FILE(GLOB MDSTRESS_SOURCES_H   include/mdstress/*.h)
FILE(GLOB MDSTRESS_SOURCES_CPP src/*.cpp)
FILE(GLOB VORO_SOURCES_HH include/voro++/*.hh)
FILE(GLOB VORO_SOURCES_CC contrib/voro++/src/*.cc)
LIST(REMOVE_ITEM VORO_SOURCES_CC ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/voro++.cc)
LIST(REMOVE_ITEM VORO_SOURCES_CC ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/v_base_wl.cc)

SET(MDSTRESS_SOURCES ${MDSTRESS_SOURCES_CPP} ${MDSTRESS_SOURCES_H} ${VORO_SOURCES_CC} ${VORO_SOURCES_HH})

INCLUDE_DIRECTORIES(include/mdstress)
INCLUDE_DIRECTORIES(include/voro++)

FIND_LIBRARY(LAPACK_LIBRARY_SYS NAMES lapack liblapack HINTS /usr/lib REQUIRED)
if (NOT LAPACK_LIBRARY_SYS)
    UNSET(LAPACK_LIBRARY_SYS-NOTFOUND)
    UNSET(LAPACK_LIBRARY_SYS)
    # check to see if we need to download LAPACK
    SET(LAPACK_URL http://www.netlib.org/lapack/lapack-3.8.0.tar.gz)
    SET(LAPACK_DOWNLOAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib/lapack-3.8.0.tar.gz)
    FILE(DOWNLOAD ${LAPACK_URL} ${LAPACK_DOWNLOAD_PATH})
    EXECUTE_PROCESS(
        COMMAND tar xaf lapack-3.8.0.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
    ADD_SUBDIRECTORY(contrib/lapack-3.8.0)
else()
    FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack liblapack HINTS /usr/lib REQUIRED)
endif()

if(MDS_COMPILEMAIN)
    list(REMOVE_ITEM MDSTRESS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/cmd_line.cc)
    add_executable(mdstress_test ${MDSTRESS_SOURCES})
    target_link_libraries(mdstress_test PRIVATE ${LAPACK_LIBRARY} ${PYTHON_LIBRARIES})
    install(TARGETS mdstress_test RUNTIME DESTINATION bin)
else(MDS_COMPILEMAIN)
    # remove main.cpp
    LIST(REMOVE_ITEM MDSTRESS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

    # attempt to find LAPACK library
    ADD_LIBRARY(mdstress SHARED ${MDSTRESS_SOURCES})
    TARGET_LINK_LIBRARIES(mdstress ${LAPACK_LIBRARY} ${PYTHON_LIBRARIES})

    # setup any targets
    INSTALL(TARGETS mdstress RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib/static)

    INSTALL(FILES ${MDSTRESS_SOURCES_H} DESTINATION include/mdstress)
    INSTALL(FILES ${VORO_SOURCES_HH} DESTINATION include/voro++)

    # only process python bindings if requested
    if(MDS_BOOSTPYTHON)
        ADD_SUBDIRECTORY(boostpython)
    endif()

    # process tensortools
    ADD_SUBDIRECTORY(tensortools)
endif(MDS_COMPILEMAIN)
