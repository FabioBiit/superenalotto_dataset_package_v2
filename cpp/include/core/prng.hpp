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

class PCG64 final : public PRNG {
public:
    explicit PCG64(std::uint64_t s = 0xCAFEBABEDEADBEEFULL);
    std::uint64_t next_u64() override;
    void seed(std::uint64_t s) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "PCG64"; }
private:
    std::array<std::uint64_t, 2> state_{};
    std::array<std::uint64_t, 2> inc_{};
};

class Philox final : public PRNG {
public:
    explicit Philox(std::uint64_t s = 0xDEADC0DEULL);
    std::uint64_t next_u64() override;
    void seed(std::uint64_t s) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "Philox"; }
private:
    std::uint64_t counter_{0};
    std::array<std::uint64_t, 2> key_{};
};

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
};

[[nodiscard]] std::unique_ptr<PRNG> make_prng(std::string_view name, std::uint64_t seed = 0);

}  // namespace se::core
