cmake_minimum_required(VERSION 3.10)
project(Parallel-Neural-Network-Inference-Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB_RECURSE SOURCES "inference/*.cpp")

add_executable(inference_engine ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(inference_engine OpenMP::OpenMP_CXX)
endif()
