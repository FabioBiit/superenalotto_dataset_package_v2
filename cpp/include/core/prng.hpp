#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace se::core {

class PRNG {
public:
    virtual ~PRNG() = default;
    virtual std::uint64_t next_u64() = 0;
    virtual double next_unit() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / (1ULL << 53));
    }
    virtual void seed(std::uint64_t s) = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

// PCG-XSL-RR-128 (Melissa O'Neill, 2014).
// 128-bit LCG state (stored as two 64-bit halves: state_[0]=low, state_[1]=high).
// 64-bit output via XOR-SHIFT and RANDOM-ROTATE.
class PCG64 final : public PRNG {
public:
    explicit PCG64(std::uint64_t s = 0xCAFEBABEDEADBEEFULL);
    std::uint64_t next_u64() override;
    void seed(std::uint64_t s) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "PCG64"; }
private:
    std::array<std::uint64_t, 2> state_{};   // {low, high}
    std::array<std::uint64_t, 2> inc_{};     // {low, high}, low bit forced to 1
};

// Philox-4x64-10 (Salmon, Moraes, Dror, Shaw, 2011).
// 4-word counter, 2-word key. Produces 4 words per evaluation; buffer the rest.
class Philox final : public PRNG {
public:
    explicit Philox(std::uint64_t s = 0xDEADC0DEULL);
    std::uint64_t next_u64() override;
    void seed(std::uint64_t s) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "Philox"; }
private:
    std::array<std::uint64_t, 4> counter_{};
    std::array<std::uint64_t, 2> key_{};
    std::array<std::uint64_t, 4> buffer_{};
    int                           buf_pos_{4};
    void refill_();
};

// ChaCha20 (Bernstein, 2008) as a CSPRNG.
// 16x32-bit state: 4 constant + 8 key + 4 counter/nonce.
// 20 rounds per block (alternating column and diagonal QRs).
class ChaCha20 final : public PRNG {
public:
    explicit ChaCha20(std::uint64_t s = 0xFACEFEEDULL);
    std::uint64_t next_u64() override;
    void seed(std::uint64_t s) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "ChaCha20"; }
private:
    std::array<std::uint32_t, 16> state_{};
    std::array<std::uint64_t, 8>  buffer_{};
    int                            buf_pos_{8};
    void refill_();
};

[[nodiscard]] std::unique_ptr<PRNG> make_prng(std::string_view name, std::uint64_t seed = 0);

}  // namespace se::core
