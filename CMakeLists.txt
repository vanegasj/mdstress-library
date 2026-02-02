cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(mdstress_test LANGUAGES CXX )
#cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
#project(mdstress_test LANGUAGES CXX CUDA)

# set the C++11 standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_LIBDIR lib)

# compiling for native, however not using fast math
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native" CACHE STRING
    "Flags used by the compiler during release builds." FORCE)
#set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-deprecated-gpu-targets -use_fast_math -restrict -Xptxas -O3,-v -Xcompiler -O3 -arch=sm_61" CACHE STRING
#    "Flags used by the compiler during release builds." FORCE)

# disabling fused multiply add, since it actually hurts performance on x86_64
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=off")

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set the boostpython option
option(MDS_BOOSTPYTHON "Enable Boost Python bindings" OFF)

set(EXT_PRECISION "EXT_DOUBLE" CACHE STRING "Precision used by exposed function calls")
set_property(CACHE EXT_PRECISION PROPERTY STRINGS EXT_DOUBLE EXT_FLOAT)
set(MDS_PRECISION "MDS_DOUBLE" CACHE STRING "Precision used by mdstresslib internally")
set_property(CACHE MDS_PRECISION PROPERTY STRINGS MDS_FLOAT MDS_DOUBLE MDS_LONG_DOUBLE)

# find the eigen library
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/contrib/eigen3")
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# get the autodiff library (I think??)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/contrib/autodiff")
find_package(autodiff)
# add_executable(mdstress_test main.cpp)

# Option to build tests
option(BUILD_MDSTRESS_TESTS "Enable building of mdstress test executables" ON)

# If tests are disabled, remove old test executables during configuration
if(NOT BUILD_MDSTRESS_TESTS)
    message(STATUS "Tests are disabled. Removing old test executables in build/test ...")

    # compute path to build/test directory
    set(TEST_DIR "${CMAKE_BINARY_DIR}/test")

    # only remove if it exists
    if(EXISTS ${TEST_DIR})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${TEST_DIR}
            RESULT_VARIABLE REMOVE_TEST_DIR_RESULT
            OUTPUT_VARIABLE REMOVE_TEST_DIR_OUTPUT
            ERROR_VARIABLE REMOVE_TEST_DIR_ERROR
        )

        if(NOT REMOVE_TEST_DIR_RESULT EQUAL 0)
            message(WARNING "Failed to remove ${TEST_DIR}: ${REMOVE_TEST_DIR_ERROR}")
        else()
            message(STATUS "Removed old test executables at ${TEST_DIR}")
        endif()
    endif()
endif()

# set our cmake modules path for tclap
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# create our mds_defines.h file in the binary build directory
configure_file(include/mdstress/mds_defines.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/mdstress/mds_defines.h @ONLY)

# find all the source files
file(GLOB MDSTRESS_SOURCES src/*.cpp src/*.cu)
file(GLOB VORO_SOURCES contrib/voro++/src/*.cc)

# the include files
file(GLOB MDSTRESS_SOURCES_H include/mdstress/*.h )
file(GLOB VORO_SOURCES_HH include/voro++/*.hh)

# remove sources
list(REMOVE_ITEM VORO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/voro++.cc)
list(REMOVE_ITEM VORO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/v_base_wl.cc)
list(REMOVE_ITEM VORO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/voro++/src/cmd_line.cc)

# add warnings etc for compiling mdstress
set_source_files_properties(
    ${MDSTRESS_SOURCES} ${MDSTRESS_SOURCES_HH}
    PROPERTIES
    COMPILE_FLAGS "-Wall -Werror -Wfatal-errors"
)

# include the headers directory
include_directories(include/mdstress include/voro++ ${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include/mdstress include/voro++)
#link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/lib64)

add_library(mdstress SHARED ${MDSTRESS_SOURCES} ${VORO_SOURCES})
#set_target_properties(mdstress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(mdstress PRIVATE Eigen3::Eigen)

target_link_libraries(mdstress PRIVATE autodiff::autodiff)

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

if(BUILD_MDSTRESS_TESTS)
    add_subdirectory(test)
endif()
