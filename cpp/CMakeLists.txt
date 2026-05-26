cmake_minimum_required(VERSION 3.24)

project(superenalotto_engine
    VERSION 0.1.0
    DESCRIPTION "High-performance reverse engineering engine for SuperEnalotto"
    LANGUAGES CXX
)

# ---------- Options ----------
option(SE_BUILD_CUDA      "Enable CUDA acceleration"        OFF)
option(SE_BUILD_PYBIND    "Build pybind11 Python bindings"  ON)
option(SE_BUILD_TESTS     "Build Catch2 unit tests"         ON)
option(SE_BUILD_CLI       "Build standalone CLI executable" ON)
option(SE_USE_OPENMP      "Enable OpenMP parallelism"       ON)

# ---------- C++ standard ----------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Output dirs ----------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------- Build type default ----------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- Compiler warnings & optimization ----------
if(MSVC)
    add_compile_options(/W3 /permissive- /Zc:__cplusplus /MP)
    add_compile_options($<$<CONFIG:Release>:/O2>)
    add_compile_options($<$<CONFIG:Release>:/Oi>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()

# ---------- Dependencies ----------
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")

# Eigen: prefer local deps/eigen, fall back to find_package
if(EXISTS "${DEPS_DIR}/eigen/Eigen/Core")
    message(STATUS "Eigen: using local copy at ${DEPS_DIR}/eigen")
    add_library(se_eigen INTERFACE)
    target_include_directories(se_eigen INTERFACE "${DEPS_DIR}/eigen")
    add_library(Eigen3::Eigen ALIAS se_eigen)
else()
    find_package(Eigen3 3.4 CONFIG REQUIRED)
endif()

# OpenMP
if(SE_USE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(STATUS "OpenMP not found, disabling parallel support")
        set(SE_USE_OPENMP OFF)
    endif()
endif()

# CUDA
if(SE_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Python + pybind11
if(SE_BUILD_PYBIND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_PROBE_RC
        )
        if(PYBIND11_PROBE_RC EQUAL 0 AND PYBIND11_CMAKE_DIR)
            message(STATUS "pybind11: using pip-installed at ${PYBIND11_CMAKE_DIR}")
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        endif()
    endif()
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Catch2: prefer local deps/catch2, fall back to find_package
if(SE_BUILD_TESTS)
    if(EXISTS "${DEPS_DIR}/catch2/CMakeLists.txt")
        message(STATUS "Catch2: using local copy at ${DEPS_DIR}/catch2")
        set(CATCH_INSTALL_DOCS    OFF CACHE BOOL "" FORCE)
        set(CATCH_INSTALL_EXTRAS  OFF CACHE BOOL "" FORCE)
        set(CATCH_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
        set(CATCH_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
        set(CATCH_BUILD_EXTRA_TESTS OFF CACHE BOOL "" FORCE)
        add_subdirectory("${DEPS_DIR}/catch2" "${CMAKE_BINARY_DIR}/_catch2-build" EXCLUDE_FROM_ALL)
        list(APPEND CMAKE_MODULE_PATH "${DEPS_DIR}/catch2/extras")
    else()
        find_package(Catch2 3 CONFIG REQUIRED)
    endif()
    enable_testing()
endif()

# ---------- Subdirectories ----------
add_subdirectory(src)

if(SE_BUILD_PYBIND)
    add_subdirectory(python)
endif()

if(SE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ---------- Summary ----------
message(STATUS "")
message(STATUS "===== SuperEnalotto Engine v${PROJECT_VERSION} =====")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP       : ${SE_USE_OPENMP}")
message(STATUS "  CUDA         : ${SE_BUILD_CUDA}")
message(STATUS "  pybind11     : ${SE_BUILD_PYBIND}")
message(STATUS "  CLI          : ${SE_BUILD_CLI}")
message(STATUS "  Tests        : ${SE_BUILD_TESTS}")
message(STATUS "===================================")
message(STATUS "")
