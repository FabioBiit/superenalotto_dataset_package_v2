pybind11_add_module(se_engine_py MODULE bindings.cpp)

target_link_libraries(se_engine_py PRIVATE se_engine)
set_target_properties(se_engine_py PROPERTIES OUTPUT_NAME "se_engine")
