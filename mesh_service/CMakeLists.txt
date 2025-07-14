cmake_minimum_required(VERSION 3.16)
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
# msgpack is header-only, no find_package needed

# Find or fetch Rerun SDK
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nvidia_mc
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /usr/local/include
)

# Source files
set(SOURCES
    src/main.cpp  # Full version with Rerun
    src/mesh_generator.cu  # Updated to use new algorithm selector
    src/gpu_octree.cu  # Still used for spatial queries
    src/shared_memory.cpp
    src/rabbitmq_consumer.cpp
    src/metrics.cpp
    src/rerun_publisher.cpp
    src/normal_estimation.cu  # Still needed for normals
    
    # New algorithm implementation
    src/simple_tsdf.cu
    src/algorithms/nvidia_marching_cubes.cu
    src/algorithm_selector.cpp
    
    # FUTURE: Uncomment these to enable Poisson/NKSR algorithms
    # src/algorithms/open3d_poisson.cpp
    # src/algorithms/nksr_client.cpp
    
    # Temporarily disabled for build
    # src/websocket_server.cpp
)

# Create executable
add_executable(mesh_service ${SOURCES})

# Set CUDA architectures (RTX 3090 is compute capability 8.6)
set_property(TARGET mesh_service PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)

# Compile options
target_compile_options(mesh_service PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo --extended-lambda>
)

# Link libraries
target_link_libraries(mesh_service PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cufft_LIBRARY}
    CGAL::CGAL
    pthread
    dl
    rt
    amqpcpp
    ev
    # msgpackc  # Using header-only version
    rerun_sdk
    stdc++fs  # C++17 filesystem
)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(mesh_service PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install target
install(TARGETS mesh_service DESTINATION bin)