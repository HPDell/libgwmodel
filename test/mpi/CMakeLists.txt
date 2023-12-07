cmake_minimum_required(VERSION 3.17)

if(ENABLE_MPI)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_HEADER_DIR})
add_link_options(${MPI_CXX_LINK_FLAGS})
add_compile_options(${MPI_CXX_COMPILE_OPTIONS})
add_definitions(${MPI_CXX_COMPILE_DEFINITIONS})
endif(ENABLE_MPI)

add_executable(testGWRBasicMpi testGWRBasicMpi.cpp ${ADDON_SOURCES})
target_link_libraries(testGWRBasicMpi PRIVATE gwmodel ${ARMADILLO_LIBRARIES} ${MPI_CXX_LIBRARIES} Catch2::Catch2WithMain)
add_test(
    NAME testGWRBasicMpi 
    COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:testGWRBasicMpi> --success --skip-benchmarks
)
