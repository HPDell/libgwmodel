cmake_minimum_required(VERSION 3.11)
project(pygwmodel)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

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
target_link_libraries(mat_interface gwmodel)
python_extension_module(mat_interface LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)

add_cython_target(name_list_interface.pyx CXX)
add_library(name_list_interface MODULE ${name_list_interface})
target_link_libraries(name_list_interface gwmodel)
python_extension_module(name_list_interface LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)

add_cython_target(simple_layer.pyx CXX)
add_library(simple_layer MODULE ${simple_layer})
target_link_libraries(simple_layer gwmodel)
python_extension_module(simple_layer LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)

add_cython_target(variable_interface.pyx CXX)
add_library(variable_interface MODULE ${variable_interface})
target_link_libraries(variable_interface gwmodel)
python_extension_module(variable_interface LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)

# add_cython_target(gwss.pyx CXX)
# add_library(gwss MODULE ${gwss})
# target_link_libraries(gwss gwmodel)
# python_extension_module(gwss LINKED_MODULES_VAR linked_module_list FORWARD_DECL_MODULES_VAR fdecl_module_list)

add_cython_target(pygwmodel.pyx CXX)
add_library(pygwmodel MODULE ${pygwmodel})
target_link_libraries(pygwmodel ${linked_module_list})
python_extension_module(pygwmodel)