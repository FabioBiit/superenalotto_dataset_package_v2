#include <catch2/catch_test_macros.hpp>
#include "core/prng.hpp"

#include <unordered_set>

using namespace se::core;

TEST_CASE("PCG64 deterministic with seed", "[prng]") {
    PCG64 a(42), b(42);
    for (int i = 0; i < 100; ++i) REQUIRE(a.next_u64() == b.next_u64());
}

TEST_CASE("PCG64 different seeds diverge", "[prng]") {
    PCG64 a(1), b(2);
    std::unordered_set<std::uint64_t> seen;
    for (int i = 0; i < 50; ++i) seen.insert(a.next_u64() ^ b.next_u64());
    REQUIRE(seen.size() > 40);
}

TEST_CASE("PRNG factory", "[prng]") {
    REQUIRE(make_prng("PCG64",    0)->name() == "PCG64");
    REQUIRE(make_prng("Philox",   0)->name() == "Philox");
    REQUIRE(make_prng("ChaCha20", 0)->name() == "ChaCha20");
    REQUIRE_THROWS(make_prng("Unknown", 0));
}

TEST_CASE("next_unit in [0, 1)", "[prng]") {
    PCG64 r(123);
    for (int i = 0; i < 1000; ++i) {
        double u = r.next_unit();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
    }
}
