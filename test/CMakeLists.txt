cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 REQUIRED)

find_package(Armadillo REQUIRED)
if(ARMADILLO_FOUND)
    add_definitions(-DARMA_USE_LAPACK -DARMA_USE_BLAS -DARMA_DONT_USE_WRAPPER)
    include_directories(${ARMADILLO_INCLUDE_DIRS})

    find_package(BLAS)
    if(NOT BLAS_FOUND)
        message(SEND_ERROR "BLAS Not Found!")
    endif(NOT BLAS_FOUND)
    
    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
        message(SEND_ERROR "LAPACK Not Found!")
    endif(NOT LAPACK_FOUND)
else(ARMADILLO_FOUND)
    message(FATAL_ERROR "Armadillo Not Found!")
endif(ARMADILLO_FOUND)

if(BUILD_SHARED_LIBS)
    add_definitions(-DGWMODEL_SHARED_LIB)
endif(BUILD_SHARED_LIBS)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${gwmodel_BINARY_DIR}/bin)

set(SAMPLE_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_definitions(-DSAMPLE_DATA_DIR="${SAMPLE_DATA_DIR}")
set(SAMPL_LONDONHP100 ${CMAKE_CURRENT_SOURCE_DIR}/data/londonhp100.cpp)

add_subdirectory(${CMAKE_SOURCE_DIR}/test/BasicGWR)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/GWSS)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/Multithread)
