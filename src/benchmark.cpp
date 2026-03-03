// Multi-stage signal pipeline benchmark: serial vs OpenMP parallel.
// Each instrument runs through rolling volatility, an EMA chain,
// spectral analysis (DFT on log-return windows), and a final score
// aggregation — enough compute to surface real parallel speedups.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <vector>

struct PipelineResult {
    double volatility = 0.0;
    double spectral_energy = 0.0;
    double ema_composite = 0.0;
    double signal_score = 0.0;
};

// Stage 1: overlapping-window rolling volatility.
double rolling_volatility(std::span<const double> prices, int window) {
    int n = static_cast<int>(prices.size());
    int stride = window / 2;
    double total_vol = 0.0;
    int count = 0;

    for (int start = 0; start + window <= n; start += stride) {
        auto win = prices.subspan(start, window);
        double mean = 0.0;
        for (double p : win) mean += p;
        mean /= window;

        double var = 0.0;
        for (double p : win) {
            double d = p - mean;
            var += d * d;
        }
        total_vol += std::sqrt(var / window);
        ++count;
    }
    return count > 0 ? total_vol / count : 0.0;
}

// Stage 2: composite EMA across six standard periods.
double ema_composite(std::span<const double> prices) {
    constexpr int periods[] = {5, 10, 20, 50, 100, 200};
    double composite = 0.0;

    for (int period : periods) {
        double alpha = 2.0 / (period + 1);
        double ema = prices[0];
        for (std::size_t t = 1; t < prices.size(); t++)
            ema = alpha * prices[t] + (1.0 - alpha) * ema;
        composite += ema;
    }
    return composite;
}

// Stage 3: DFT-based spectral energy on non-overlapping log-return windows.
// Heavy on transcendentals (log, sin, cos) — deliberately CPU-bound.
double spectral_energy(std::span<const double> prices, int window) {
    int n = static_cast<int>(prices.size());
    if (n < window + 1) return 0.0;

    constexpr int NUM_BINS = 12;
    constexpr double TWO_PI = 2.0 * std::numbers::pi;
    double total = 0.0;
    int num_windows = (n - 1) / window;

    for (int w = 0; w < num_windows; w++) {
        int base = w * window;
        for (int k = 0; k < NUM_BINS; k++) {
            double re = 0.0, im = 0.0;
            double freq = TWO_PI * k / window;
            for (int j = 0; j < window; j++) {
                double lr = std::log(prices[base + j + 1] / prices[base + j]);
                double angle = freq * j;
                re += lr * std::cos(angle);
                im += lr * std::sin(angle);
            }
            total += re * re + im * im;
        }
    }
    return total;
}

PipelineResult compute_pipeline(std::span<const double> prices) {
    PipelineResult r;
    r.volatility = rolling_volatility(prices, 64);
    r.ema_composite = ema_composite(prices);
    r.spectral_energy = spectral_energy(prices, 64);
    r.signal_score = r.ema_composite * r.volatility + std::sqrt(r.spectral_energy);
    return r;
}

int main() {
    constexpr int N = 2000;
    constexpr int TICKS = 5000;
    constexpr int WARMUP = 3;
    constexpr int REPS = 10;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.01);
    std::uniform_real_distribution<double> base_price(20.0, 500.0);
    std::vector<std::vector<double>> data(N, std::vector<double>(TICKS));
    for (auto& series : data) {
        series[0] = base_price(rng);
        for (int t = 1; t < TICKS; t++)
            series[t] = series[t - 1] * (1.0 + noise(rng));
    }
    std::vector<PipelineResult> results(N);

    auto bench = [&](bool parallel) {
        std::vector<long long> timings;
        for (int r = 0; r < WARMUP + REPS; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (parallel) {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < N; i++)
                    results[i] = compute_pipeline(data[i]);
            } else {
                for (int i = 0; i < N; i++)
                    results[i] = compute_pipeline(data[i]);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r >= WARMUP)
                timings.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                        .count());
        }
        std::ranges::sort(timings);
        return timings[timings.size() / 2];
    };

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "=== Signal Pipeline Benchmark ===\n";
    std::cout << "Instruments:  " << N << "  |  Ticks each: " << TICKS << "\n";
    std::cout << "Pipeline:     rolling vol -> EMA chain (6 periods) "
                 "-> spectral DFT (12 bins) -> score\n";
    std::cout << "Threads:      " << num_threads << "\n";
    std::cout << "Warmup:       " << WARMUP
              << " runs  |  Timed: " << REPS << " runs\n\n";

    long long s = bench(false);
    long long p = bench(true);

    std::cout << "Serial (median):   " << s << " us\n";
    std::cout << "Parallel (median): " << p << " us\n";
    std::cout << "Speedup:           " << static_cast<double>(s) / p << "x\n";

    return 0;
}
