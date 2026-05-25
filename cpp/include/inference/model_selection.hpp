#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace se::inference {

struct ModelMetrics {
    std::string model_name;
    double log_likelihood{0.0};
    int    n_parameters{0};
    int    n_observations{0};
    double aic{0.0};
    double bic{0.0};
    double waic{0.0};
    double loo_cv{0.0};
};

[[nodiscard]] ModelMetrics compute_metrics(
    const std::string& name,
    double log_likelihood,
    int n_parameters,
    int n_observations,
    const Eigen::VectorXd& pointwise_log_likelihood = {});

[[nodiscard]] std::vector<ModelMetrics> rank_models(std::vector<ModelMetrics> models,
                                                       const std::string& criterion = "BIC");

}  // namespace se::inference
