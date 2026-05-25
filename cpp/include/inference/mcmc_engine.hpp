#pragma once

#include <Eigen/Dense>
#include <functional>

namespace se::inference {

using LogPosterior = std::function<double(const Eigen::VectorXd&)>;

struct MHConfig {
    int    n_iterations = 50'000;
    int    n_burn_in    = 10'000;
    double step_size    = 0.1;
    bool   adapt        = true;
    std::uint64_t seed  = 17;
};

struct MHResult {
    Eigen::MatrixXd chain;
    Eigen::VectorXd log_post_chain;
    double          acceptance_rate{0.0};
};

[[nodiscard]] MHResult metropolis_hastings(LogPosterior log_post,
                                             const Eigen::VectorXd& init,
                                             const MHConfig& cfg = {});

}  // namespace se::inference
