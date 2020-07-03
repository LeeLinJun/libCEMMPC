cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cem LANGUAGES CXX CUDA)
find_package(CUDA QUIET REQUIRED)

include_directories(
    ./include
)

set(CARTPOLE_SRC
    src/cartpole/cartpole.cu
    src/cartpole/cartpole_parallel.cu
    src/acrobot/acrobot.cu
    src/main.cpp)

add_executable(demo ${CARTPOLE_SRC})
# add_executable(cartpole_diff ${CARTPOLE_DIFF_SRC})
