cmake_minimum_required(VERSION 3.9)
project(hnsw_index)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -fopenmp")

include_directories(PYTHON_INCLUDE_DIR)
include_directories(src/third_party)

add_executable(tests src/hnsw/tests.cpp)
add_executable(benchmarks src/hnsw/benchmarks.cpp)