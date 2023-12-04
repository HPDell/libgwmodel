cmake_minimum_required(VERSION 3.17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${libgwmodel_BINARY_DIR}/bin)

if(ENABLE_OpenMP)
find_package(OpenMP)
if(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    if(ENABLE_OpenMP)
        add_definitions(-DENABLE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
        include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
    endif(ENABLE_OpenMP)
endif(OpenMP_FOUND AND OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
endif()

if(ENABLE_MPI)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_HEADER_DIR})
add_link_options(${MPI_CXX_LINK_FLAGS})
add_compile_options(${MPI_CXX_COMPILE_OPTIONS})
add_definitions(${MPI_CXX_COMPILE_DEFINITIONS})
endif(ENABLE_MPI)

find_package(GSL REQUIRED)
if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
else()
    message(FATAL "GSL not found!")
endif()


include_directories(
    ${LIBGWMODEL_INCLUDE_DIR} 
    ${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp
    ${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp/utils
    ${LIBGWMODEL_INCLUDE_DIR}/gwmodelcuda
)

add_definitions(-DM_PI=3.14159265358979323846 -DM_PI_2=1.57079632679489661923 -DDOUBLE_EPS=1e-8)
if(NOT MSVC)
    add_definitions(-DDBL_MAX=__DBL_MAX__)    
endif(NOT MSVC)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-D_DEBUG)
endif()


set(SOURCES_CXX
    gwmodelpp/spatialweight/BandwidthWeight.cpp
    gwmodelpp/spatialweight/CRSDistance.cpp
    gwmodelpp/spatialweight/OneDimDistance.cpp
    gwmodelpp/spatialweight/Distance.cpp
    gwmodelpp/spatialweight/DMatDistance.cpp
    gwmodelpp/spatialweight/MinkwoskiDistance.cpp
    gwmodelpp/spatialweight/SpatialWeight.cpp
    gwmodelpp/spatialweight/Weight.cpp
    gwmodelpp/spatialweight/CRSSTDistance.cpp

    gwmodelpp/BandwidthSelector.cpp
    gwmodelpp/VariableForwardSelector.cpp
    gwmodelpp/SpatialAlgorithm.cpp
    gwmodelpp/SpatialMonoscaleAlgorithm.cpp
    gwmodelpp/SpatialMultiscaleAlgorithm.cpp
    gwmodelpp/GWRBase.cpp
    gwmodelpp/GWRBasic.cpp
    gwmodelpp/GWSS.cpp
    gwmodelpp/GWPCA.cpp
    gwmodelpp/GWDR.cpp
    gwmodelpp/GWRMultiscale.cpp
    gwmodelpp/GWRRobust.cpp
    gwmodelpp/GWRGeneralized.cpp
    gwmodelpp/BinomialModel.cpp
    gwmodelpp/PoissonModel.cpp
    gwmodelpp/GeneralizedLinearModel.cpp
    gwmodelpp/GWRScalable.cpp
    gwmodelpp/GWRLocalCollinearity.cpp
    gwmodelpp/GTWR.cpp
    gwmodelpp/GWDA.cpp
)

set(SOURCES_C
    gwmodel.cpp
)

set(HEADERS_CXX
    ../include/gwmodelpp/spatialweight/BandwidthWeight.h
    ../include/gwmodelpp/spatialweight/CRSDistance.h
    ../include/gwmodelpp/spatialweight/OneDimDistance.h
    ../include/gwmodelpp/spatialweight/Distance.h
    ../include/gwmodelpp/spatialweight/DMatDistance.h
    ../include/gwmodelpp/spatialweight/MinkwoskiDistance.h
    ../include/gwmodelpp/spatialweight/SpatialWeight.h
    ../include/gwmodelpp/spatialweight/Weight.h
    ../include/gwmodelpp/spatialweight/CRSSTDistance.h

    ../include/gwmodelpp/Algorithm.h
    ../include/gwmodelpp/BandwidthSelector.h
    ../include/gwmodelpp/VariableForwardSelector.h
    ../include/gwmodelpp/SpatialAlgorithm.h
    ../include/gwmodelpp/SpatialMonoscaleAlgorithm.h
    ../include/gwmodelpp/SpatialMultiscaleAlgorithm.h
    ../include/gwmodelpp/GWRBase.h
    ../include/gwmodelpp/GWRBasic.h
    ../include/gwmodelpp/GWSS.h
    ../include/gwmodelpp/GWPCA.h
    ../include/gwmodelpp/GWDR.h
    ../include/gwmodelpp/GWRMultiscale.h
    ../include/gwmodelpp/Logger.h
    ../include/gwmodelpp/GWRRobust.h
    ../include/gwmodelpp/GWRGeneralized.h
    ../include/gwmodelpp/BinomialModel.h
    ../include/gwmodelpp/LinearModel.h
    ../include/gwmodelpp/PoissonModel.h
    ../include/gwmodelpp/GeneralizedLinearModel.h
    ../include/gwmodelpp/GWRScalable.h
    ../include/gwmodelpp/GWRLocalCollinearity.h
    ../include/gwmodelpp/GTWR.h
    ../include/gwmodelpp/GWDA.h
)

set(HEADERS_C
    ../include/gwmodel.h
)

set(HEADERS_ALL ${HEADERS_CXX} ${HEADERS_C})
set(SOURCES_ALL ${SOURCES_CXX})

if(ENABLE_CUDA)
    set(HEADERS_CUDA
        ../include/gwmodelpp/utils/CudaUtils.h
        ../include/gwmodelpp/utils/cumat.hpp
        ../include/gwmodelpp/spatialweight/cuda/BandwidthWeightKernel.h
        ../include/gwmodelpp/spatialweight/cuda/CRSDistanceKernel.h
    )
    set(SOURCES_CUDA
        gwmodelpp/utils/CudaUtils.cpp
        gwmodelpp/utils/cumat.cpp
        gwmodelpp/spatialweight/cuda/BandwidthWeightKernel.cu
        gwmodelpp/spatialweight/cuda/CRSDistanceKernel.cu
    )
    list(PREPEND HEADERS_ALL ${HEADERS_CUDA})
    list(PREPEND SOURCES_ALL ${SOURCES_CUDA})
endif(ENABLE_CUDA)

add_library(gwmodel STATIC ${HEADERS_ALL} ${SOURCES_ALL})
set_property(TARGET gwmodel PROPERTY POSITION_INDEPENDENT_CODE ON)


message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    if(MSVC)
        target_link_libraries(gwmodel
            CUDA::cudart_static
            CUDA::cublas
        )
    else(MSVC)
        target_link_libraries(gwmodel
            CUDA::cudart_static
            CUDA::cublas_static
        )
        target_compile_options(gwmodel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --compiler-options
            -Werror
        >)
    endif(MSVC)
endif(ENABLE_CUDA AND CUDAToolkit_FOUND)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
        target_compile_options(gwmodel PRIVATE -Wmost -Werror)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
        target_compile_options(gwmodel PRIVATE /W4 /wd4819 /WX)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        target_compile_options(gwmodel PRIVATE -Wall -Werror -O0)
    endif()
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
        target_compile_options(gwmodel PRIVATE -Wmost)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
        target_compile_options(gwmodel PRIVATE /W4 /wd4819)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        target_compile_options(gwmodel PRIVATE -Wall)
    endif()
endif()

target_link_libraries(gwmodel
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    GSL::gsl
    GSL::gslcblas
)

if(OpenMP_FOUND)
    target_link_libraries(gwmodel
        ${OpenMP_C_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
    )
endif(OpenMP_FOUND)

if(ENABLE_MPI AND MPI_FOUND)
    target_link_libraries(gwmodel
        ${MPI_CXX_LIBRARIES}
    )
endif()

if(USE_CUDA_SHARED)
    set(HEADERS_CUDA_SHARED
        ../include/gwmodelcuda/StdTelegram.h
        ../include/gwmodelcuda/GWRBasicGpuTask.h
        ../include/gwmodelcuda/IGWRBasicGpuTask.h
    )
    set(SOURCES_CUDA_SHARED
        gwmodelcuda/StdTelegram.cpp
        gwmodelcuda/IGWRBasicGpuTask.cpp
        gwmodelcuda/GWRBasicGpuTask.cpp
    )
    add_definitions(-DCREATDLL_EXPORTS)
    add_library(gwmodelcuda SHARED ${HEADERS_CUDA_SHARED} ${SOURCES_CUDA_SHARED})
    target_link_libraries(gwmodelcuda
        PRIVATE gwmodel ${ARMADILLO_LIBRARIES}
    )
    set_property(TARGET gwmodelcuda PROPERTY POSITION_INDEPENDENT_CODE ON)
endif(USE_CUDA_SHARED)

install(TARGETS gwmodel
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY ../include/ DESTINATION include)