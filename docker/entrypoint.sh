#!/bin/bash

# Build ORB_SLAM3 dependencies if they haven't been built yet
if [ ! -d "/app/third_party/ORB_SLAM3/Thirdparty/DBoW2/build" ]; then
    cd /app/third_party/ORB_SLAM3/Thirdparty/DBoW2 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-maybe-uninitialized" .. && \
    make -j8
fi

if [ ! -d "/app/third_party/ORB_SLAM3/Thirdparty/g2o/build" ]; then
    cd /app/third_party/ORB_SLAM3/Thirdparty/g2o && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-maybe-uninitialized" .. && \
    make -j8
fi

# Execute the passed command or default to bash
exec "$@" || exec "/bin/bash" 