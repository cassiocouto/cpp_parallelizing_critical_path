// Volatility benchmark: serial vs OpenMP parallel with guided scheduling.
// Reproduces the main benchmark from the article.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

double compute_volatility(const std::vector<double>& prices) {
    double mean = 0.0;
    for (double p : prices) mean += p;
    mean /= static_cast<double>(prices.size());

    double variance = 0.0;
    for (double p : prices) variance += (p - mean) * (p - mean);
    return std::sqrt(variance / static_cast<double>(prices.size()));
}

int main() {
    const int N = 1000;
    const int TICKS = 1000;
    const int WARMUP = 3;
    const int REPS = 10;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.01);
    std::vector<std::vector<double>> data(N, std::vector<double>(TICKS));
    for (auto& series : data) {
        series[0] = 100.0;
        for (int t = 1; t < TICKS; t++)
            series[t] = series[t - 1] * (1.0 + noise(rng));
    }
    std::vector<double> results(N);

    auto bench = [&](bool parallel) {
        std::vector<long long> timings;
        for (int r = 0; r < WARMUP + REPS; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (parallel) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < N; i++)
                    results[i] = compute_volatility(data[i]);
            } else {
                for (int i = 0; i < N; i++)
                    results[i] = compute_volatility(data[i]);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r >= WARMUP)
                timings.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                        .count());
        }
        std::sort(timings.begin(), timings.end());
        return timings[timings.size() / 2];
    };

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "=== Volatility Benchmark ===\n";
    std::cout << "Instruments: " << N << "  |  Ticks each: " << TICKS << "\n";
    std::cout << "Threads:     " << num_threads << "\n";
    std::cout << "Warmup:      " << WARMUP << " runs  |  Timed: " << REPS << " runs\n\n";

    long long s = bench(false);
    long long p = bench(true);

    std::cout << "Serial (median):   " << s << " us\n";
    std::cout << "Parallel (median): " << p << " us\n";
    std::cout << "Speedup:           " << static_cast<double>(s) / p << "x\n";

    return 0;
}
