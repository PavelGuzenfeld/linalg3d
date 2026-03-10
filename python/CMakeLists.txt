cmake_minimum_required(VERSION 3.25)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
project(linalg3d_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python (nanobind requires 'Python' not 'Python3')
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find or fetch nanobind
find_package(nanobind QUIET)
if(NOT nanobind_FOUND)
    include(FetchContent)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.4.0
        SYSTEM EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(nanobind)
endif()

# Find or fetch gcem
include(FetchContent)
set(FETCHCONTENT_QUIET ON)
find_package(gcem QUIET)
if(NOT gcem_FOUND)
    FetchContent_Declare(gcem
        GIT_REPOSITORY https://github.com/kthohr/gcem.git
        GIT_TAG master
        SYSTEM EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(gcem)
endif()

# Build the nanobind module
nanobind_add_module(_core src/bindings.cpp)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(_core PRIVATE gcem)

# Enable native SIMD
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_core PRIVATE -march=native)
endif()

install(TARGETS _core LIBRARY DESTINATION linalg3d)
