#include "core/prng.hpp"

#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
  #pragma intrinsic(_umul128)
#endif

namespace se::core {

namespace {

// 64x64 -> 128 multiply: returns (lo, hi).
inline void mul_64x64_128(std::uint64_t a, std::uint64_t b,
                           std::uint64_t& lo, std::uint64_t& hi) noexcept {
#if defined(__SIZEOF_INT128__)
    const __uint128_t r = static_cast<__uint128_t>(a) * b;
    lo = static_cast<std::uint64_t>(r);
    hi = static_cast<std::uint64_t>(r >> 64);
#elif defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_AMD64))
    lo = _umul128(a, b, &hi);
#else
    const std::uint64_t al = a & 0xFFFFFFFFULL, ah = a >> 32;
    const std::uint64_t bl = b & 0xFFFFFFFFULL, bh = b >> 32;
    const std::uint64_t ll = al * bl;
    const std::uint64_t lh = al * bh;
    const std::uint64_t hl = ah * bl;
    const std::uint64_t hh = ah * bh;
    const std::uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    lo = (ll & 0xFFFFFFFFULL) | (mid << 32);
    hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
}

// 128-bit add with carry, in-place: state += addend.
inline void add128_inplace(std::uint64_t* s_lo, std::uint64_t* s_hi,
                            std::uint64_t a_lo, std::uint64_t a_hi) noexcept {
    const std::uint64_t old_lo = *s_lo;
    *s_lo = old_lo + a_lo;
    const std::uint64_t carry = (*s_lo < old_lo) ? 1 : 0;
    *s_hi = *s_hi + a_hi + carry;
}

// state = state * MULT, low 128 bits.
inline void mul128_inplace(std::uint64_t* s_lo, std::uint64_t* s_hi,
                            std::uint64_t m_lo, std::uint64_t m_hi) noexcept {
    std::uint64_t ll_lo, ll_hi;
    mul_64x64_128(*s_lo, m_lo, ll_lo, ll_hi);
    const std::uint64_t add_hi = (*s_lo) * m_hi + (*s_hi) * m_lo;
    *s_lo = ll_lo;
    *s_hi = ll_hi + add_hi;
}

inline std::uint64_t rotr64(std::uint64_t v, unsigned r) noexcept {
    return (v >> r) | (v << ((-static_cast<std::int64_t>(r)) & 63));
}

inline std::uint32_t rotl32(std::uint32_t v, unsigned r) noexcept {
    return (v << r) | (v >> ((-static_cast<std::int32_t>(r)) & 31));
}

// PCG64 multiplier constant (128-bit): 0x2360ED051FC65DA44385DF649FCCF645
constexpr std::uint64_t PCG_MULT_LO = 0x4385DF649FCCF645ULL;
constexpr std::uint64_t PCG_MULT_HI = 0x2360ED051FC65DA4ULL;

// SplitMix64 (Steele/Lea) for seeding diverse PRNGs from a 64-bit seed.
inline std::uint64_t splitmix64(std::uint64_t& s) noexcept {
    s += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

// =====================================================================
// PCG-XSL-RR-128
// =====================================================================
PCG64::PCG64(std::uint64_t s) { seed(s); }

void PCG64::seed(std::uint64_t s) {
    std::uint64_t sm = s;
    state_[0] = 0;
    state_[1] = 0;
    inc_[0]   = (splitmix64(sm) << 1) | 1ULL;   // must be odd
    inc_[1]   = splitmix64(sm);
    (void)next_u64();
    const std::uint64_t init_lo = splitmix64(sm);
    const std::uint64_t init_hi = splitmix64(sm);
    add128_inplace(&state_[0], &state_[1], init_lo, init_hi);
    (void)next_u64();
}

std::uint64_t PCG64::next_u64() {
    // LCG: state = state * MULT + INC
    mul128_inplace(&state_[0], &state_[1], PCG_MULT_LO, PCG_MULT_HI);
    add128_inplace(&state_[0], &state_[1], inc_[0], inc_[1]);
    // XSL-RR: XOR shift, then random rotate
    const std::uint64_t xored = state_[0] ^ state_[1];
    const unsigned       rot   = static_cast<unsigned>(state_[1] >> 58);  // top 6 bits
    return rotr64(xored, rot);
}

// =====================================================================
// Philox-4x64-10
// =====================================================================
namespace {
constexpr std::uint64_t PHILOX_M0 = 0xD2E7470EE14C6C93ULL;
constexpr std::uint64_t PHILOX_M1 = 0xCA5A826395121157ULL;
constexpr std::uint64_t PHILOX_C0 = 0x9E3779B97F4A7C15ULL;   // golden ratio
constexpr std::uint64_t PHILOX_C1 = 0xBB67AE8584CAA73BULL;   // sqrt(3) - 1
}

Philox::Philox(std::uint64_t s) { seed(s); }

void Philox::seed(std::uint64_t s) {
    std::uint64_t sm = s;
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = 0;
    counter_[3] = 0;
    key_[0] = splitmix64(sm);
    key_[1] = splitmix64(sm);
    buf_pos_ = 4;
}

void Philox::refill_() {
    std::uint64_t x0 = counter_[0], x1 = counter_[1], x2 = counter_[2], x3 = counter_[3];
    std::uint64_t k0 = key_[0],     k1 = key_[1];

    for (int r = 0; r < 10; ++r) {
        std::uint64_t lo0, hi0, lo1, hi1;
        mul_64x64_128(PHILOX_M0, x0, lo0, hi0);
        mul_64x64_128(PHILOX_M1, x2, lo1, hi1);
        const std::uint64_t nx0 = hi1 ^ x1 ^ k0;
        const std::uint64_t nx1 = lo1;
        const std::uint64_t nx2 = hi0 ^ x3 ^ k1;
        const std::uint64_t nx3 = lo0;
        x0 = nx0; x1 = nx1; x2 = nx2; x3 = nx3;
        if (r < 9) {
            k0 += PHILOX_C0;
            k1 += PHILOX_C1;
        }
    }
    buffer_[0] = x0;
    buffer_[1] = x1;
    buffer_[2] = x2;
    buffer_[3] = x3;
    buf_pos_ = 0;

    // Increment 256-bit counter
    if (++counter_[0] == 0)
        if (++counter_[1] == 0)
            if (++counter_[2] == 0) ++counter_[3];
}

std::uint64_t Philox::next_u64() {
    if (buf_pos_ >= 4) refill_();
    return buffer_[buf_pos_++];
}

// =====================================================================
// ChaCha20 (Bernstein, 2008) as CSPRNG
// =====================================================================
namespace {

inline void chacha_qr(std::uint32_t& a, std::uint32_t& b,
                       std::uint32_t& c, std::uint32_t& d) noexcept {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

constexpr std::uint32_t CHACHA_CONST[4] = {
    0x61707865,  // "expa"
    0x3320646e,  // "nd 3"
    0x79622d32,  // "2-by"
    0x6b206574   // "te k"
};

}  // namespace

ChaCha20::ChaCha20(std::uint64_t s) { seed(s); }

void ChaCha20::seed(std::uint64_t s) {
    std::uint64_t sm = s;
    state_[0] = CHACHA_CONST[0];
    state_[1] = CHACHA_CONST[1];
    state_[2] = CHACHA_CONST[2];
    state_[3] = CHACHA_CONST[3];
    for (int i = 4; i < 12; ++i) {
        state_[i] = static_cast<std::uint32_t>(splitmix64(sm));
    }
    state_[12] = 0;                                       // block counter (low)
    state_[13] = 0;                                       // block counter (high)
    state_[14] = static_cast<std::uint32_t>(splitmix64(sm));   // nonce[0]
    state_[15] = static_cast<std::uint32_t>(splitmix64(sm));   // nonce[1]
    buf_pos_ = 8;
}

void ChaCha20::refill_() {
    std::uint32_t s[16];
    std::memcpy(s, state_.data(), sizeof(s));

    for (int i = 0; i < 10; ++i) {
        // Column rounds
        chacha_qr(s[0], s[4], s[ 8], s[12]);
        chacha_qr(s[1], s[5], s[ 9], s[13]);
        chacha_qr(s[2], s[6], s[10], s[14]);
        chacha_qr(s[3], s[7], s[11], s[15]);
        // Diagonal rounds
        chacha_qr(s[0], s[5], s[10], s[15]);
        chacha_qr(s[1], s[6], s[11], s[12]);
        chacha_qr(s[2], s[7], s[ 8], s[13]);
        chacha_qr(s[3], s[4], s[ 9], s[14]);
    }
    for (int i = 0; i < 16; ++i) s[i] += state_[i];

    // Pack 16 x u32 into 8 x u64 (little-endian).
    for (int i = 0; i < 8; ++i) {
        buffer_[i] = static_cast<std::uint64_t>(s[2 * i])
                   | (static_cast<std::uint64_t>(s[2 * i + 1]) << 32);
    }
    buf_pos_ = 0;

    // Increment 64-bit block counter (state_[12], state_[13]).
    if (++state_[12] == 0) ++state_[13];
}

std::uint64_t ChaCha20::next_u64() {
    if (buf_pos_ >= 8) refill_();
    return buffer_[buf_pos_++];
}

// =====================================================================
// Factory
// =====================================================================
std::unique_ptr<PRNG> make_prng(std::string_view name, std::uint64_t seed) {
    if (name == "PCG64")    return std::make_unique<PCG64>(seed ? seed : 0xCAFEBABEDEADBEEFULL);
    if (name == "Philox")   return std::make_unique<Philox>(seed ? seed : 0xDEADC0DEULL);
    if (name == "ChaCha20") return std::make_unique<ChaCha20>(seed ? seed : 0xFACEFEEDULL);
    throw std::invalid_argument("unknown PRNG: " + std::string(name));
}

}  // namespace se::core
