#!/usr/bin/env bash
set -euo pipefail
# Use GCC for build (Clang 18 + libstdc++ has incomplete std::expected support)
CC=gcc CXX=g++ cmake -B build -DCMAKE_CXX_STANDARD=23 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# Install FetchContent deps system-wide so clang-tidy can find them when analyzing headers
if [ -d build/_deps/gcem-src ]; then
    cmake --install build/_deps/gcem-build 2>/dev/null || \
        cp -r build/_deps/gcem-src/include/* /usr/local/include/ 2>/dev/null || true
fi
