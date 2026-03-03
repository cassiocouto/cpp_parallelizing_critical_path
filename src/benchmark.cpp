// Multi-stage signal pipeline benchmark: serial vs OpenMP parallel.
//
// Simulates a per-instrument feature-compute stage in a trading system.
// Each instrument runs through four stages — rolling volatility, an EMA
// chain, spectral analysis (DFT on log-return windows), and a final
// score aggregation.  The pipeline is deliberately heavy on
// transcendentals (log, sin, cos, sqrt) so that serial runtime lands in
// the seconds range, making the parallel speedup unambiguous.
//
// Every instrument is independent — the textbook case for OpenMP
// parallel-for with guided scheduling.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <span>        // C++20: lightweight, non-owning view over contiguous data
#include <vector>

// Aggregated output of the four pipeline stages for a single instrument.
struct PipelineResult {
    double volatility = 0.0;
    double spectral_energy = 0.0;
    double ema_composite = 0.0;
    double signal_score = 0.0;
};

// ---------------------------------------------------------------------------
// Stage 1 — Rolling volatility
//
// Slides overlapping windows (50% overlap) across the price series and
// computes standard deviation in each window.  Returns the average
// volatility across all windows.  The overlap means adjacent windows
// share half their data, which is typical in real rolling-window analytics.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Stage 2 — Composite EMA
//
// Computes an exponential moving average at each of six standard periods
// (5, 10, 20, 50, 100, 200) and sums them into a single composite value.
// In a real system these would feed downstream models; here the six full
// passes over the price series add meaningful per-instrument work.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Stage 3 — Spectral energy via DFT on log-return windows
//
// Splits the series into non-overlapping windows, computes log-returns
// within each window, then runs a partial DFT (12 frequency bins) to
// extract spectral energy.  This is the heaviest stage: every element
// triggers log(), sin(), and cos() — hundreds of thousands of
// transcendental calls per instrument.
// ---------------------------------------------------------------------------
double spectral_energy(std::span<const double> prices, int window) {
    int n = static_cast<int>(prices.size());
    if (n < window + 1) return 0.0;

    constexpr int NUM_BINS = 12;
    constexpr double TWO_PI = 2.0 * std::numbers::pi;   // C++20 <numbers>
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
            total += re * re + im * im;   // |X[k]|^2, power at bin k
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// Stage 4 — Aggregate into a final signal score
//
// Chains the three stages and combines them into a single score.
// In production this would be a model input; here it just makes sure
// the compiler can't optimize away any of the earlier work.
// ---------------------------------------------------------------------------
PipelineResult compute_pipeline(std::span<const double> prices) {
    PipelineResult r;
    r.volatility = rolling_volatility(prices, 64);
    r.ema_composite = ema_composite(prices);
    r.spectral_energy = spectral_energy(prices, 64);
    r.signal_score = r.ema_composite * r.volatility + std::sqrt(r.spectral_energy);
    return r;
}

// ---------------------------------------------------------------------------

int main() {
    constexpr int N = 2000;        // number of instruments in the universe
    constexpr int TICKS = 5000;    // price ticks per instrument
    constexpr int WARMUP = 3;      // untimed runs to warm caches / JIT
    constexpr int REPS = 10;       // timed runs; we report the median

    // -- Data generation: geometric random walk per instrument -------------
    // Each series starts at a random base price and evolves with 1%
    // daily noise — close enough to real tick data for benchmarking.
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

    // -- Benchmark harness -------------------------------------------------
    // Runs WARMUP+REPS iterations, discards the first WARMUP, returns
    // the median of the remaining REPS.  Median is more robust than mean
    // against occasional OS scheduling jitter.
    auto bench = [&](bool parallel) {
        std::vector<long long> timings;
        for (int r = 0; r < WARMUP + REPS; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (parallel) {
                // guided scheduling: large chunks first, shrinking over time.
                // Good default when per-instrument work is roughly uniform
                // but not identical (price series lengths could vary).
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
        std::ranges::sort(timings);            // C++20 ranges
        return timings[timings.size() / 2];    // median
    };

    // Query the actual thread-team size (respects OMP_NUM_THREADS env var).
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
