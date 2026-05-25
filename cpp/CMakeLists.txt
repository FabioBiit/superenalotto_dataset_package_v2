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
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /MP)
    add_compile_options($<$<CONFIG:Release>:/O2 /Oi /GL>)
    add_link_options($<$<CONFIG:Release>:/LTCG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
    add_compile_options($<$<CONFIG:Release>:-O3 -march=native -flto>)
endif()

# ---------- Dependencies ----------
find_package(Eigen3 3.4 CONFIG REQUIRED)

if(SE_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(SE_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(SE_BUILD_PYBIND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

if(SE_BUILD_TESTS)
    find_package(Catch2 3 CONFIG REQUIRED)
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
