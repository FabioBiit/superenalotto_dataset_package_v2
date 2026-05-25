#include "core/prng.hpp"

#include <cstring>
#include <random>
#include <stdexcept>

namespace se::core {

PCG64::PCG64(std::uint64_t s) { seed(s); }

void PCG64::seed(std::uint64_t s) {
    state_[0] = s ^ 0x9E3779B97F4A7C15ULL;
    state_[1] = s + 0xBB67AE8584CAA73BULL;
    inc_[0]   = 0xDA3E39CB94B95BDBULL;
    inc_[1]   = 0xC2B2AE3D27D4EB4FULL;
    (void)next_u64();
}

std::uint64_t PCG64::next_u64() {
    state_[0] = state_[0] * 6364136223846793005ULL + (inc_[0] | 1ULL);
    state_[1] = state_[1] * 1442695040888963407ULL + (inc_[1] | 1ULL);
    std::uint64_t xorshifted = ((state_[0] >> 18u) ^ state_[0]) >> 27u;
    std::uint64_t rot = state_[0] >> 59u;
    std::uint64_t out = (xorshifted >> rot) | (xorshifted << ((-static_cast<std::int64_t>(rot)) & 63));
    return out ^ state_[1];
}

Philox::Philox(std::uint64_t s) { seed(s); }

void Philox::seed(std::uint64_t s) {
    counter_ = 0;
    key_[0]  = s;
    key_[1]  = s ^ 0xCBF29CE484222325ULL;
}

std::uint64_t Philox::next_u64() {
    ++counter_;
    std::uint64_t x = counter_ * 0x9E3779B97F4A7C15ULL + key_[0];
    std::uint64_t y = (counter_ ^ key_[1]) * 0xBF58476D1CE4E5B9ULL;
    x ^= (y >> 30u);
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31u);
}

ChaCha20::ChaCha20(std::uint64_t s) { seed(s); }

void ChaCha20::seed(std::uint64_t s) {
    std::mt19937_64 rng(s);
    for (auto& v : state_) v = static_cast<std::uint32_t>(rng());
    for (auto& v : buffer_) v = rng();
    buf_pos_ = 8;
}

std::uint64_t ChaCha20::next_u64() {
    if (buf_pos_ >= 8) {
        std::mt19937_64 rng(state_[0] | (std::uint64_t{state_[1]} << 32));
        for (auto& v : buffer_) v = rng();
        ++state_[0];
        buf_pos_ = 0;
    }
    return buffer_[buf_pos_++];
}

std::unique_ptr<PRNG> make_prng(std::string_view name, std::uint64_t seed) {
    if (name == "PCG64")    return std::make_unique<PCG64>(seed ? seed : 0xCAFEBABEDEADBEEFULL);
    if (name == "Philox")   return std::make_unique<Philox>(seed ? seed : 0xDEADC0DEULL);
    if (name == "ChaCha20") return std::make_unique<ChaCha20>(seed ? seed : 0xFACEFEEDULL);
    throw std::invalid_argument("unknown PRNG: " + std::string(name));
}

}  // namespace se::core
