cmake_minimum_required(VERSION 3.1)

project(testMultithread LANGUAGES CXX)

set(SOURCES static.cpp)
if(BUILD_SHARED_LIBS)
set(SOURCES shared.cpp)
endif(BUILD_SHARED_LIBS)

add_executable(testMultithread ${SOURCES} ${SAMPL_LONDONHP100})
target_link_libraries(testMultithread PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)

add_test(
    NAME testMultithread 
    COMMAND $<TARGET_FILE:testMultithread> --success
)
