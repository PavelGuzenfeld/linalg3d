#!/usr/bin/env bash
set -euo pipefail
# Use GCC for build (Clang 18 + libstdc++ has incomplete std::expected support)
CC=gcc CXX=g++ cmake -B build -DCMAKE_CXX_STANDARD=23 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# Install FetchContent headers into build/sysinclude so clang-tidy can find
# them when analyzing headers independently (compile_commands.json has
# -isystem paths but clang-tidy doesn't use them for standalone header analysis).
mkdir -p build/sysinclude
for dep in gcem fmt doctest; do
    src="build/_deps/${dep}-src/include"
    [ -d "$src" ] && cp -r "$src"/* build/sysinclude/ 2>/dev/null || true
done
# Also copy doctest single-header (lives at root, not include/)
[ -f "build/_deps/doctest-src/doctest/doctest.h" ] && \
    mkdir -p build/sysinclude/doctest && \
    cp build/_deps/doctest-src/doctest/doctest.h build/sysinclude/doctest/ 2>/dev/null || true
