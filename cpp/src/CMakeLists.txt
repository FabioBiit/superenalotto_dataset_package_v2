file(GLOB_RECURSE SE_CORE_SRC       core/*.cpp)
file(GLOB_RECURSE SE_MODELS_SRC     models/*.cpp)
file(GLOB_RECURSE SE_INFERENCE_SRC  inference/*.cpp)
file(GLOB_RECURSE SE_GENERATION_SRC generation/*.cpp)

add_library(se_engine STATIC
    ${SE_CORE_SRC}
    ${SE_MODELS_SRC}
    ${SE_INFERENCE_SRC}
    ${SE_GENERATION_SRC}
)

target_include_directories(se_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(se_engine PUBLIC
    Eigen3::Eigen
)

if(SE_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(se_engine PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(se_engine PUBLIC SE_HAS_OPENMP=1)
endif()

if(SE_BUILD_CUDA)
    file(GLOB_RECURSE SE_CUDA_SRC core/*.cu models/*.cu)
    if(SE_CUDA_SRC)
        target_sources(se_engine PRIVATE ${SE_CUDA_SRC})
        set_target_properties(se_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(se_engine PUBLIC CUDA::cudart)
        target_compile_definitions(se_engine PUBLIC SE_HAS_CUDA=1)
    endif()
endif()

if(SE_BUILD_CLI)
    add_executable(se_cli main.cpp)
    target_link_libraries(se_cli PRIVATE se_engine)
endif()
