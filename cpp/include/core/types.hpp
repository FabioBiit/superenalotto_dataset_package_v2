#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace se::core {

constexpr int N_MAIN = 6;
constexpr int N_MAX  = 90;
constexpr std::uint64_t C_6_90 = 622'614'630ULL;

enum class Operatore : std::uint8_t { SisalApax, SisalFlutter };

struct Draw {
    std::chrono::year_month_day date{};
    int contest_number{0};
    std::array<int, N_MAIN> main{};
    int jolly{0};
    int superstar{0};
    std::optional<int>    winners_6{};
    std::optional<double> prize_6{};
    Operatore operatore{Operatore::SisalApax};
    bool regime_4week{false};
    bool regime_post_DM214{false};
    std::string source_url{};
};

using DrawSet = std::vector<Draw>;

[[nodiscard]] inline bool is_valid_number(int n) noexcept {
    return n >= 1 && n <= N_MAX;
}

[[nodiscard]] inline bool is_valid_main(const std::array<int, N_MAIN>& m) noexcept {
    bool seen[N_MAX + 1] = {false};
    for (int v : m) {
        if (!is_valid_number(v) || seen[v]) return false;
        seen[v] = true;
    }
    return true;
}

}  // namespace se::core
