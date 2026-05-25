#pragma once

#include "core/types.hpp"
#include <filesystem>

namespace se::core {

struct LoadOptions {
    bool strict_validation = true;
    bool skip_header = true;
};

class DataLoader {
public:
    [[nodiscard]] static DrawSet load_csv(const std::filesystem::path& path,
                                          const LoadOptions& opts = {});

    [[nodiscard]] static std::size_t count_invalid(const DrawSet& draws) noexcept;
};

}  // namespace se::core
