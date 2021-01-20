cmake_minimum_required(VERSION 3.1)

project(testBasicGWR LANGUAGES CXX)

set(SOURCES static.cpp)
if(BUILD_SHARED_LIBS)
set(SOURCES shared.cpp)
endif(BUILD_SHARED_LIBS)

add_executable(testBasicGWR ${SOURCES} ${SAMPL_LONDONHP100})
target_link_libraries(testBasicGWR PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)

add_test(
    NAME testBasicGWR 
    COMMAND $<TARGET_FILE:testBasicGWR> --success
)
