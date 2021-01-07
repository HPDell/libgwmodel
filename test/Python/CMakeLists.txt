cmake_minimum_required(VERSION 3.1)
project(testPython LANGUAGES CXX)

add_custom_target(testPythonDeps ALL 
    VERBATIM 
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E
        make_directory
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        touch
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/pygwmodel/__init__.py
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        $<TARGET_FILE:gwss>
        $<TARGET_FILE:mat_interface>
        $<TARGET_FILE:name_list_interface>
        $<TARGET_FILE:simple_layer>
        $<TARGET_FILE:spatial_weight>
        $<TARGET_FILE:variable_interface>
        $<TARGET_FILE:regression_diagnostic>
        $<TARGET_FILE:gwr_basic>
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/pygwmodel
    # COMMAND ${CMAKE_COMMAND} -E
    #     copy_if_different
    #     ${CMAKE_CURRENT_SOURCE_DIR}/testGWSS.py
    #     ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>
)
