cmake_minimum_required(VERSION 3.22)

project (dpconv)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra")

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

include_directories(include)
include_directories(util)
include_directories(types)
include_directories(algorithms)

file(GLOB INCLUDE_FILES
  "include/QueryGraph.hpp"
  "include/BitSet.hpp"
  "include/Util.hpp"
  "include/Common.hpp"
  "include/SubsetConvolution.hpp"
  "util/BenchmarkRunner.hpp"
)

file(GLOB THIRD_PARTY_FILES
  "third-party/*.hpp"
)

file(GLOB ALGORITHM_FILES
  "algorithms/DPconv_old.cpp"
  "algorithms/DPsub.cpp"
  "algorithms/DPccp.cpp"
  "algorithms/DPconv.cpp"
)

file(GLOB TYPE_FILES "types/StaticMinMaxRingElement.hpp")

# Collect all.
file(GLOB ALL_FILES ${INCLUDE_FILES} ${ALGORITHM_FILES} ${TYPE_FILES})

add_executable(bench ${ALL_FILES} bench.cc)

add_executable(clique_bench ${ALL_FILES} clique_bench.cc)