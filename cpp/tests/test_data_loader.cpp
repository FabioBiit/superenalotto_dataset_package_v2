#include <catch2/catch_test_macros.hpp>
#include "core/types.hpp"

using namespace se::core;

TEST_CASE("is_valid_number", "[types]") {
    REQUIRE(is_valid_number(1));
    REQUIRE(is_valid_number(45));
    REQUIRE(is_valid_number(90));
    REQUIRE_FALSE(is_valid_number(0));
    REQUIRE_FALSE(is_valid_number(91));
    REQUIRE_FALSE(is_valid_number(-5));
}

TEST_CASE("is_valid_main", "[types]") {
    REQUIRE(is_valid_main({1, 2, 3, 4, 5, 6}));
    REQUIRE(is_valid_main({90, 1, 45, 22, 78, 13}));
    REQUIRE_FALSE(is_valid_main({1, 2, 3, 4, 5, 5}));    // duplicate
    REQUIRE_FALSE(is_valid_main({0, 2, 3, 4, 5, 6}));    // out of range
    REQUIRE_FALSE(is_valid_main({1, 2, 3, 4, 5, 91}));   // out of range
}
