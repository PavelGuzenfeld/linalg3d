cmake_minimum_required(VERSION 3.22)
project(linalg3d VERSION 0.1.0 LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(cmake-library REQUIRED)
find_package(fmt REQUIRED)
find_package(gcem REQUIRED)

# Runner library
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

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

# Test executable
add_executable(${PROJECT_NAME}_test test/${PROJECT_NAME}_test.cpp)
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME} fmt gcem)

# Ensure asserts stay enabled in tests, even in Release mode
set_warnings_and_errors(${PROJECT_NAME}_test)

# Export the target
install(
    TARGETS ${PROJECT_NAME}_test
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION bin
)

# Handle "Manually-specified variables were not used by the project" warning.
unset(CATKIN_INSTALL_INTO_PREFIX_ROOT)
unset(CATKIN_SYMLINK_INSTALL)
