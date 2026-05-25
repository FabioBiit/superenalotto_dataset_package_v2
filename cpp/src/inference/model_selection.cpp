#include "inference/model_selection.hpp"

#include <algorithm>
#include <cmath>

namespace se::inference {

ModelMetrics compute_metrics(
    const std::string& name,
    double log_likelihood,
    int n_parameters,
    int n_observations,
    const Eigen::VectorXd& pointwise_log_likelihood) {

    ModelMetrics m;
    m.model_name     = name;
    m.log_likelihood = log_likelihood;
    m.n_parameters   = n_parameters;
    m.n_observations = n_observations;

    m.aic = 2.0 * n_parameters - 2.0 * log_likelihood;
    m.bic = std::log(static_cast<double>(std::max(n_observations, 1))) * n_parameters - 2.0 * log_likelihood;

    if (pointwise_log_likelihood.size() > 0) {
        const double lpd = pointwise_log_likelihood.sum();
        const double mean = pointwise_log_likelihood.mean();
        double var = 0.0;
        for (int i = 0; i < pointwise_log_likelihood.size(); ++i) {
            const double d = pointwise_log_likelihood(i) - mean;
            var += d * d;
        }
        var /= std::max<int>(pointwise_log_likelihood.size() - 1, 1);
        m.waic = -2.0 * (lpd - var);
        m.loo_cv = m.waic;  // Pareto-smoothed LOO postponed to Phase E.4
    }
    return m;
}

std::vector<ModelMetrics> rank_models(std::vector<ModelMetrics> models,
                                       const std::string& criterion) {
    auto key = [&](const ModelMetrics& m) {
        if (criterion == "AIC")  return m.aic;
        if (criterion == "BIC")  return m.bic;
        if (criterion == "WAIC") return m.waic;
        if (criterion == "LOO")  return m.loo_cv;
        return m.bic;
    };
    std::sort(models.begin(), models.end(),
              [&](const ModelMetrics& a, const ModelMetrics& b) { return key(a) < key(b); });
    return models;
}

}  // namespace se::inference
