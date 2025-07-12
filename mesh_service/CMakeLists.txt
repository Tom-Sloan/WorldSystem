cmake_minimum_required(VERSION 3.14)
project(mesh_service LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)

# Find packages
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(OpenMP)

# Find or fetch Rerun SDK
include(FetchContent)
FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /usr/local/include
)

# Source files
set(SOURCES
    src/main.cpp
    src/simple_mesh_generator.cu
    src/shared_memory.cpp
    src/rabbitmq_consumer.cpp
    src/metrics.cpp
    src/rerun_publisher.cpp
    # Temporarily disabled for build
    # src/websocket_server.cpp
    # src/mesh_generator.cu
    # src/poisson_reconstruction.cpp
    # src/marching_cubes.cu
    # src/normal_estimation.cu
    # src/marching_cubes_tables.cu
)

# Create executable
add_executable(mesh_service ${SOURCES})

# Set CUDA architectures (RTX 3090 is compute capability 8.6)
set_property(TARGET mesh_service PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)

# Compile options
target_compile_options(mesh_service PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

# Link libraries
target_link_libraries(mesh_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    CGAL::CGAL
    pthread
    dl
    rt
    amqpcpp
    ev
    msgpackc
    rerun_sdk
)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(mesh_service OpenMP::OpenMP_CXX)
endif()

# Install target
install(TARGETS mesh_service DESTINATION bin)