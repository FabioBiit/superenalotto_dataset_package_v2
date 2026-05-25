add_executable(se_tests
    test_data_loader.cpp
    test_prng.cpp
    test_stats.cpp
    test_model_selection.cpp
    test_monte_carlo.cpp
    test_generation.cpp
)

target_link_libraries(se_tests PRIVATE
    se_engine
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(se_tests)
