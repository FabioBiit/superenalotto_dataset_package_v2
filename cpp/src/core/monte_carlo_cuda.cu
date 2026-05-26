// CUDA kernels for SuperEnalotto Monte Carlo simulations.
// Each thread performs N_per_thread complete draws (sampling 6 from 90).
// Per-thread frequency vectors are atomic-added to a single global VectorXd.
//
// Build only when SE_BUILD_CUDA=ON. Linked into se_engine via CMake when CUDA is found.

#include "core/types.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <Eigen/Dense>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace se::core::cuda {

namespace {

constexpr int N_MAX_DEV  = 90;
constexpr int N_MAIN_DEV = 6;

#define SE_CUDA_CHECK(expr)                                                                   \
    do {                                                                                       \
        cudaError_t _err = (expr);                                                             \
        if (_err != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));  \
        }                                                                                      \
    } while (0)

__device__ inline int weighted_pick_no_repl_dev(curandStatePhilox4_32_10_t& rng,
                                                  double* w, bool* picked) {
    // Sequential weighted sampling without replacement among N_MAX_DEV balls.
    double total = 0.0;
    #pragma unroll
    for (int i = 0; i < N_MAX_DEV; ++i)
        if (!picked[i]) total += w[i];

    const double r = static_cast<double>(curand_uniform_double(&rng)) * total;
    double cum = 0.0;
    int chosen = -1;
    #pragma unroll
    for (int i = 0; i < N_MAX_DEV; ++i) {
        if (picked[i]) continue;
        cum += w[i];
        if (chosen < 0 && cum >= r) chosen = i;
    }
    if (chosen < 0) {
        for (int i = N_MAX_DEV - 1; i >= 0; --i) if (!picked[i]) { chosen = i; break; }
    }
    picked[chosen] = true;
    return chosen;
}

__global__ void kernel_uniform_freq(std::uint64_t seed,
                                      int n_draws_total,
                                      unsigned long long* global_counts) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    const int per_thread = (n_draws_total + nthreads - 1) / nthreads;
    const int my_start = tid * per_thread;
    const int my_end   = min(my_start + per_thread, n_draws_total);

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);

    __shared__ unsigned int local_counts[N_MAX_DEV];
    if (threadIdx.x < N_MAX_DEV) local_counts[threadIdx.x] = 0;
    __syncthreads();

    int deck[N_MAX_DEV];
    for (int d = my_start; d < my_end; ++d) {
        #pragma unroll
        for (int i = 0; i < N_MAX_DEV; ++i) deck[i] = i + 1;
        for (int i = 0; i < N_MAIN_DEV; ++i) {
            const unsigned int r = curand(&rng);
            const int j = i + static_cast<int>(r % static_cast<unsigned int>(N_MAX_DEV - i));
            int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
            atomicAdd(&local_counts[deck[i] - 1], 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x < N_MAX_DEV)
        atomicAdd(&global_counts[threadIdx.x],
                  static_cast<unsigned long long>(local_counts[threadIdx.x]));
}

__global__ void kernel_weighted_freq(std::uint64_t seed,
                                       const double* __restrict__ weights,
                                       int n_draws_total,
                                       unsigned long long* global_counts) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    const int per_thread = (n_draws_total + nthreads - 1) / nthreads;
    const int my_start = tid * per_thread;
    const int my_end   = min(my_start + per_thread, n_draws_total);

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);

    __shared__ unsigned int local_counts[N_MAX_DEV];
    if (threadIdx.x < N_MAX_DEV) local_counts[threadIdx.x] = 0;
    __syncthreads();

    double w[N_MAX_DEV];
    bool   picked[N_MAX_DEV];

    for (int d = my_start; d < my_end; ++d) {
        #pragma unroll
        for (int i = 0; i < N_MAX_DEV; ++i) { w[i] = weights[i]; picked[i] = false; }
        for (int k = 0; k < N_MAIN_DEV; ++k) {
            const int chosen = weighted_pick_no_repl_dev(rng, w, picked);
            atomicAdd(&local_counts[chosen], 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x < N_MAX_DEV)
        atomicAdd(&global_counts[threadIdx.x],
                  static_cast<unsigned long long>(local_counts[threadIdx.x]));
}

}  // anonymous namespace

// =====================================================================
// Public API (called from monte_carlo.cpp when cfg.use_cuda && SE_HAS_CUDA)
// =====================================================================

Eigen::VectorXd simulate_uniform_frequency_cuda(int n_draws, std::uint64_t seed) {
    if (n_draws <= 0) throw std::invalid_argument("n_draws must be > 0");

    unsigned long long* d_counts = nullptr;
    SE_CUDA_CHECK(cudaMalloc(&d_counts, N_MAX_DEV * sizeof(unsigned long long)));
    SE_CUDA_CHECK(cudaMemset(d_counts, 0, N_MAX_DEV * sizeof(unsigned long long)));

    const int block = 128;
    const int target_threads = 1 << 14;          // 16K threads
    const int grid  = (target_threads + block - 1) / block;
    kernel_uniform_freq<<<grid, block>>>(seed, n_draws, d_counts);
    SE_CUDA_CHECK(cudaGetLastError());
    SE_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned long long> h_counts(N_MAX_DEV, 0);
    SE_CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                              N_MAX_DEV * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    SE_CUDA_CHECK(cudaFree(d_counts));

    Eigen::VectorXd freq(N_MAX_DEV);
    const double denom = static_cast<double>(n_draws) * N_MAIN_DEV;
    for (int i = 0; i < N_MAX_DEV; ++i)
        freq(i) = static_cast<double>(h_counts[i]) / denom;
    return freq;
}

Eigen::VectorXd simulate_weighted_frequency_cuda(const double* weights_host,
                                                   int n_draws,
                                                   std::uint64_t seed) {
    if (n_draws <= 0) throw std::invalid_argument("n_draws must be > 0");

    double*             d_weights = nullptr;
    unsigned long long* d_counts  = nullptr;
    SE_CUDA_CHECK(cudaMalloc(&d_weights, N_MAX_DEV * sizeof(double)));
    SE_CUDA_CHECK(cudaMalloc(&d_counts,  N_MAX_DEV * sizeof(unsigned long long)));
    SE_CUDA_CHECK(cudaMemcpy(d_weights, weights_host, N_MAX_DEV * sizeof(double),
                              cudaMemcpyHostToDevice));
    SE_CUDA_CHECK(cudaMemset(d_counts, 0, N_MAX_DEV * sizeof(unsigned long long)));

    const int block = 128;
    const int target_threads = 1 << 13;          // 8K threads (smaller for weighted = more work)
    const int grid  = (target_threads + block - 1) / block;
    kernel_weighted_freq<<<grid, block>>>(seed, d_weights, n_draws, d_counts);
    SE_CUDA_CHECK(cudaGetLastError());
    SE_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned long long> h_counts(N_MAX_DEV, 0);
    SE_CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                              N_MAX_DEV * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    SE_CUDA_CHECK(cudaFree(d_weights));
    SE_CUDA_CHECK(cudaFree(d_counts));

    Eigen::VectorXd freq(N_MAX_DEV);
    const double denom = static_cast<double>(n_draws) * N_MAIN_DEV;
    for (int i = 0; i < N_MAX_DEV; ++i)
        freq(i) = static_cast<double>(h_counts[i]) / denom;
    return freq;
}

#undef SE_CUDA_CHECK

}  // namespace se::core::cuda
