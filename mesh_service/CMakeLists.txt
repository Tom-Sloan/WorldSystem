cmake_minimum_required(VERSION 3.16)
project(mesh_service LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Find dependencies
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

find_package(Open3D REQUIRED)

# RabbitMQ-C (pkg-config)
pkg_check_modules(RABBITMQ REQUIRED librabbitmq)

# msgpack-c (pkg-config)
pkg_check_modules(MSGPACK REQUIRED msgpack)

# Find or fetch Rerun SDK
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Add executable (include shared_memory.cpp)
add_executable(mesh_service src/main.cpp src/shared_memory.cpp)

# Include directories
target_include_directories(mesh_service PRIVATE include)

# Link libraries
target_link_libraries(mesh_service
    PRIVATE
    rerun_sdk  # Rerun SDK
    ${RABBITMQ_LIBRARIES}
    ${MSGPACK_LIBRARIES}
    Threads::Threads
    dl  # For dlopen if needed
    rt  # For shared memory
)

target_include_directories(mesh_service PRIVATE ${RABBITMQ_INCLUDE_DIRS} ${MSGPACK_INCLUDE_DIRS})

target_link_libraries(mesh_service PRIVATE Open3D::Open3D)  