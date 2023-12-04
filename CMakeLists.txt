cmake_minimum_required(VERSION 3.17)
project(libgwmodel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH cmake)

option(ENABLE_OpenMP "Determines whether OpemMP support should be built" ON)
option(ENABLE_CUDA "Determines whether CUDA support should be built" OFF)
option(ENABLE_MPI "Determines whether MPI support should be built" OFF)
option(WITH_TESTS "Determines whether to build and run tests" ON)

if(ENABLE_CUDA)
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)
    option(USE_CUDA_SHARED "Determines whether to build shared libraries to call CUDA" OFF)

    find_package(CUDAToolkit REQUIRED)
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDAToolkit Not Found!")
    else(NOT CUDAToolkit_FOUND)
        message(STATUS "Found CUDAToolkit at: ${CUDAToolkit_INCLUDE_DIRS}")
    endif(NOT CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

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

if(MSVC)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

set(LIBGWMODEL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)

if(WITH_TESTS)
enable_testing()
add_subdirectory(test)
endif()
