#pragma once

#include "core/types.hpp"
#include <Eigen/Dense>

namespace se::models {

struct HawkesParams {
    double mu    = 0.1;
    double alpha = 0.5;
    double beta  = 1.0;
};

struct HawkesFit {
    HawkesParams params{};
    double log_likelihood{0.0};
    bool   converged{false};
};

class Hawkes {
public:
    [[nodiscard]] static HawkesFit fit(const Eigen::VectorXd& event_times, int max_iter = 500);
    [[nodiscard]] static double    intensity_at(double t,
                                                  const Eigen::VectorXd& past_events,
                                                  const HawkesParams& p);
};

}  // namespace se::models
