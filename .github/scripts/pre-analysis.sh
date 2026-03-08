#!/usr/bin/env bash
set -euo pipefail
# Use GCC for build (Clang 18 + libstdc++ has incomplete std::expected support)
CC=gcc CXX=g++ cmake -B build -DCMAKE_CXX_STANDARD=23 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
