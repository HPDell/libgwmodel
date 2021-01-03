cmake_minimum_required(VERSION 3.1)
project(gwmodel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Weather to build a shared library" OFF) 
if(BUILD_SHARED_LIBS)
    message(STATUS "The library can only be used by C-style interfaces.")
endif(BUILD_SHARED_LIBS)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

if(BUILD_SHARED_LIBS)
    option(WITH_PYTHON "Whether to build python extension" OFF)
    if(WITH_PYTHON)
        add_subdirectory(${CMAKE_SOURCE_DIR}/python)
    endif(WITH_PYTHON)
endif(BUILD_SHARED_LIBS)


enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/test)
add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)
add_test(
    NAME testGWSS 
    COMMAND $<TARGET_FILE:testGWSS> --success
)
add_test(
    NAME testMultithread 
    COMMAND $<TARGET_FILE:testMultithread> --success
)

install(TARGETS gwmodel
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

if(WITH_PYTHON)
    install(TARGETS pygwmodel gwss mat_interface name_list_interface simple_layer spatial_weight variable_interface
            LIBRARY DESTINATION py)
endif(WITH_PYTHON)
