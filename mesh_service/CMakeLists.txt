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

# Open3D support (required for Poisson reconstruction)
option(USE_OPEN3D "Enable Open3D for Poisson reconstruction and normal estimation" ON)
if(USE_OPEN3D)
    find_package(Open3D QUIET)
    if(Open3D_FOUND)
        message(STATUS "Open3D found - enabling Poisson reconstruction and high-quality normal estimation")
        add_definitions(-DHAS_OPEN3D)
    else()
        message(WARNING "Open3D requested but not found - Poisson reconstruction will not be available")
    endif()
endif()

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
    src/configuration_manager.cpp  # New configuration system
    
    # New algorithm implementation
    src/simple_tsdf.cu
    src/algorithms/nvidia_marching_cubes.cu
    src/algorithm_selector.cpp
    
    # Normal providers
    src/normal_provider_factory.cpp
    src/normal_providers/camera_based_normal_provider.cu
    
    # Algorithm implementations
    # src/algorithms/nksr_client.cpp  # Future: NKSR integration
    
    # Temporarily disabled for build
    # src/websocket_server.cpp
)

# Conditionally add Open3D-dependent sources
if(Open3D_FOUND)
    list(APPEND SOURCES 
        src/normal_providers/open3d_normal_provider.cpp
        src/algorithms/open3d_poisson.cpp
    )
endif()

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

# Open3D support
if(Open3D_FOUND)
    target_link_libraries(mesh_service PRIVATE ${Open3D_LIBRARIES})
    target_include_directories(mesh_service PRIVATE ${Open3D_INCLUDE_DIRS})
endif()

# Install target
install(TARGETS mesh_service DESTINATION bin)