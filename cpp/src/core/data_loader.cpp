#include "core/data_loader.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace se::core {

namespace {

constexpr auto FLUTTER_YEAR  = std::chrono::year{2022};
constexpr auto FLUTTER_MONTH = std::chrono::August;
constexpr auto FLUTTER_DAY   = std::chrono::day{4};

constexpr auto FOURWEEK_YEAR  = std::chrono::year{2023};
constexpr auto FOURWEEK_MONTH = std::chrono::July;
constexpr auto FOURWEEK_DAY   = std::chrono::day{7};

constexpr auto DM214_YEAR  = std::chrono::year{2026};
constexpr auto DM214_MONTH = std::chrono::January;
constexpr auto DM214_DAY   = std::chrono::day{27};

[[nodiscard]] std::chrono::year_month_day parse_iso_date(std::string_view s) {
    int y{0}, m{0}, d{0};
    if (s.size() < 10) throw std::runtime_error("invalid date: " + std::string(s));
    auto p1 = std::from_chars(s.data(),     s.data() + 4,  y);
    auto p2 = std::from_chars(s.data() + 5, s.data() + 7,  m);
    auto p3 = std::from_chars(s.data() + 8, s.data() + 10, d);
    if (p1.ec != std::errc{} || p2.ec != std::errc{} || p3.ec != std::errc{})
        throw std::runtime_error("invalid date: " + std::string(s));
    return {std::chrono::year{y}, std::chrono::month{static_cast<unsigned>(m)},
            std::chrono::day{static_cast<unsigned>(d)}};
}

[[nodiscard]] int parse_int(std::string_view s) {
    int v{0};
    auto p = std::from_chars(s.data(), s.data() + s.size(), v);
    if (p.ec != std::errc{}) throw std::runtime_error("invalid int: " + std::string(s));
    return v;
}

[[nodiscard]] std::vector<std::string_view> split(std::string_view line, char sep) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == sep) {
            out.emplace_back(line.data() + start, i - start);
            start = i + 1;
        }
    }
    return out;
}

[[nodiscard]] bool date_geq(const std::chrono::year_month_day& a,
                              const std::chrono::year_month_day& b) {
    return std::chrono::sys_days{a} >= std::chrono::sys_days{b};
}

}  // namespace

DrawSet DataLoader::load_csv(const std::filesystem::path& path, const LoadOptions& opts) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open: " + path.string());

    DrawSet draws;
    draws.reserve(3000);

    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (first && opts.skip_header) { first = false; continue; }
        first = false;

        auto cols = split(line, ',');
        if (cols.size() < 10) {
            if (opts.strict_validation) throw std::runtime_error("malformed row: " + line);
            continue;
        }

        Draw d;
        d.date = parse_iso_date(cols[0]);
        d.contest_number = parse_int(cols[1]);
        for (int i = 0; i < N_MAIN; ++i) d.main[i] = parse_int(cols[2 + i]);
        d.jolly     = parse_int(cols[8]);
        d.superstar = parse_int(cols[9]);
        if (cols.size() >= 11 && !cols[10].empty())  d.winners_6 = parse_int(cols[10]);
        if (cols.size() >= 12 && !cols[11].empty())  d.prize_6   = std::stod(std::string(cols[11]));
        if (cols.size() >= 13)                        d.source_url = std::string(cols[12]);

        const std::chrono::year_month_day flutter{FLUTTER_YEAR, FLUTTER_MONTH, FLUTTER_DAY};
        const std::chrono::year_month_day fourweek{FOURWEEK_YEAR, FOURWEEK_MONTH, FOURWEEK_DAY};
        const std::chrono::year_month_day dm214{DM214_YEAR, DM214_MONTH, DM214_DAY};

        d.operatore           = date_geq(d.date, flutter)  ? Operatore::SisalFlutter : Operatore::SisalApax;
        d.regime_4week        = date_geq(d.date, fourweek);
        d.regime_post_DM214   = date_geq(d.date, dm214);

        if (opts.strict_validation && !is_valid_main(d.main)) {
            throw std::runtime_error("invalid main numbers in row: " + line);
        }
        draws.push_back(std::move(d));
    }
    std::sort(draws.begin(), draws.end(), [](const Draw& a, const Draw& b) {
        return std::chrono::sys_days{a.date} < std::chrono::sys_days{b.date};
    });
    return draws;
}

std::size_t DataLoader::count_invalid(const DrawSet& draws) noexcept {
    std::size_t n = 0;
    for (const auto& d : draws) {
        if (!is_valid_main(d.main)) ++n;
        else if (!is_valid_number(d.jolly) || !is_valid_number(d.superstar)) ++n;
    }
    return n;
}

}  // namespace se::core
