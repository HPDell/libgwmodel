cmake_minimum_required(VERSION 3.11)
project(pygwmodel)

find_package(PythonInterp 3)
find_package(Cython REQUIRED)
find_package(PythonExtensions REQUIRED)

set(CY_SOURCE 
    ${CMAKE_CURRENT_SOURCE_DIR}/gwss.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_interface.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/name_list_interface.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_layer.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/spatial_weight.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/variable_interface.pyx
)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

add_definitions(-DGWMODEL_SHARED_LIB)
if(MSVC)
    add_definitions(-DCREATE_DLL_EXPORTS)
endif(MSVC)

add_cython_target(mat_interface.pyx CXX)
add_library(mat_interface MODULE ${mat_interface})
target_link_libraries(mat_interface gwmodel LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)
python_extension_module(mat_interface)

add_cython_target(gwss.pyx CXX)
add_library(gwss MODULE ${gwss})
target_link_libraries(gwss gwmodel LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)
python_extension_module(gwss)

add_cython_target(pygwmodel.pyx CXX)
add_library(pygwmodel MODULE ${pygwmodel})
target_link_libraries(pygwmodel ${linked_module_list})