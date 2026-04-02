cmake_minimum_required(VERSION 3.25)

# Allow FetchContent deps with old cmake_minimum_required (e.g. gcem)
if(POLICY CMP0178)
    cmake_policy(SET CMP0178 OLD)
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

project(linalg3d VERSION 0.3.2 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
function(linalg3d_set_warnings target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
            -Wconversion -Wsign-conversion -Wnull-dereference
            -Wdouble-promotion -Wformat=2 -Wshadow
            -Wuninitialized -Wunreachable-code -Wunused
        )
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE
                -Wduplicated-cond -Wduplicated-branches
                -Wlogical-op -Wuseless-cast
            )
        endif()
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    endif()
endfunction()

# Dependencies: use system packages if available, otherwise fetch from source
include(FetchContent)
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
set(FETCHCONTENT_QUIET ON)

find_package(gcem QUIET)
if(NOT gcem_FOUND)
    FetchContent_Declare(gcem GIT_REPOSITORY https://github.com/kthohr/gcem.git GIT_TAG master
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(gcem)
endif()

find_package(fmt QUIET)
if(NOT fmt_FOUND)
    FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 11.1.4
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(fmt)
endif()

# Header-only library
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Transitive dependencies — consumers get gcem automatically
target_link_libraries(${PROJECT_NAME} INTERFACE gcem)

# Install public headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
)

# Export the target
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION include
)

# Install the export file
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Configure package files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Install the package configuration files
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Tests and benchmarks (only when built standalone)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    FetchContent_Declare(doctest GIT_REPOSITORY https://github.com/doctest/doctest.git GIT_TAG v2.4.11
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(doctest)

    FetchContent_Declare(nanobench GIT_REPOSITORY https://github.com/martinus/nanobench.git GIT_TAG v4.3.11
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(nanobench)

    FetchContent_Declare(eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG 3.4.0
                         SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_Populate(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")

    enable_testing()

    add_executable(${PROJECT_NAME}_test test/${PROJECT_NAME}_test.cpp)
    target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME} fmt gcem doctest::doctest)
    linalg3d_set_warnings(${PROJECT_NAME}_test)
    add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)

    # Enable native SIMD for tests and benchmarks
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${PROJECT_NAME}_test PRIVATE -march=native)
    endif()

    # Benchmarks
    add_executable(${PROJECT_NAME}_bench bench/${PROJECT_NAME}_bench.cpp)
    target_link_libraries(${PROJECT_NAME}_bench PRIVATE ${PROJECT_NAME} gcem nanobench::nanobench Eigen3::Eigen)
    linalg3d_set_warnings(${PROJECT_NAME}_bench)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${PROJECT_NAME}_bench PRIVATE -march=native)
    endif()
endif()

# Handle "Manually-specified variables were not used by the project" warning.
unset(CATKIN_INSTALL_INTO_PREFIX_ROOT)
unset(CATKIN_SYMLINK_INSTALL)
