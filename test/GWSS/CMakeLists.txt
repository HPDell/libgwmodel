cmake_minimum_required(VERSION 3.1)

project(testGWSS LANGUAGES CXX)

set(SOURCES static.cpp)
if(BUILD_SHARED_LIBS)
set(SOURCES shared.cpp)
endif(BUILD_SHARED_LIBS)

add_executable(testGWSS ${SOURCES} ${SAMPL_LONDONHP100})
target_link_libraries(testGWSS PRIVATE gwmodel ${ARMADILLO_LIBRARIES} Catch2::Catch2)

add_test(
    NAME testGWSS 
    COMMAND $<TARGET_FILE:testGWSS> --success
)
