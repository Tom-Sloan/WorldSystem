#!/bin/bash
cd /app/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j1 VERBOSE=1 2>&1 | tail -100