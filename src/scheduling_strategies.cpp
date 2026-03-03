// scheduling_strategies.cpp — Static vs dynamic vs guided scheduling.
//
// OpenMP's default (static) scheduling divides iterations into equal-sized
// chunks assigned to threads at compile time.  That's optimal when every
// iteration costs roughly the same.  In trading systems, instruments are
// rarely homogeneous: liquid large-caps have dense tick histories while
// illiquid names are sparse.  This example uses a heterogeneous universe
// (200–5000 ticks per instrument) to show why scheduling strategy matters.
//
//   static  — equal chunks upfront.  Fast threads idle while slow ones
//             finish their oversized chunk.
//   dynamic — each thread grabs the next available iteration when it's
//             free.  Good load balancing, but per-iteration scheduling
//             overhead adds up.
//   guided  — starts with large chunks and shrinks them over time.
//             Balances load without the per-iteration overhead of dynamic.
//
// Article section: "Scheduling Strategies: Don't Leave Performance on the Table"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <vector>

#include "market_data.h"

// ---------------------------------------------------------------------------
// Per-instrument workload: log-return DFT + 4-period EMA chain.
//
// Cost scales with the length of the price series, which varies across
// instruments in the heterogeneous universe.  This uneven cost is what
// makes scheduling strategy visible in the timings.
// ---------------------------------------------------------------------------
double heavy_compute(const std::vector<double>& prices) {
    int n = static_cast<int>(prices.size());
    if (n < 2) return 0.0;

    double result = 0.0;

    // Spectral energy: DFT over non-overlapping windows of log-returns.
    const int window = std::min(64, n - 1);
    const int NUM_BINS = 8;
    int num_windows = (n - 1) / window;
    for (int w = 0; w < num_windows; w++) {
        int base = w * window;
        for (int k = 0; k < NUM_BINS; k++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < window; j++) {
                double lr = std::log(prices[base + j + 1] / prices[base + j]);
                double angle = 2.0 * std::numbers::pi * k * j / window;
                re += lr * std::cos(angle);
                im += lr * std::sin(angle);
            }
            result += re * re + im * im;
        }
    }

    // EMA chain: four standard periods, each a full pass over the series.
    for (int period : {5, 10, 20, 50}) {
        double alpha = 2.0 / (period + 1);
        double ema = prices[0];
        for (int t = 1; t < n; t++)
            ema = alpha * prices[t] + (1.0 - alpha) * ema;
        result += ema;
    }

    return result;
}

int main() {
    const int NUM_INSTRUMENTS = 1000;
    const int WARMUP = 3;
    const int REPS = 10;

    // Heterogeneous universe: tick counts range from 200 to 5000.
    MarketUniverse universe;
    universe.generate_heterogeneous(NUM_INSTRUMENTS);

    // Collect tick-count distribution for the header output.
    std::vector<int> sizes(NUM_INSTRUMENTS);
    for (int i = 0; i < NUM_INSTRUMENTS; i++)
        sizes[i] = static_cast<int>(universe.price_series[i].size());
    std::sort(sizes.begin(), sizes.end());

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "=== Scheduling Strategies: Static vs Dynamic vs Guided ===\n";
    std::cout << "Instruments: " << NUM_INSTRUMENTS << " (heterogeneous)\n";
    std::cout << "Tick range:  [" << sizes.front() << ", " << sizes.back()
              << "]  median=" << sizes[sizes.size() / 2] << "\n";
    std::cout << "Threads:     " << num_threads << "\n";
    std::cout << "Warmup:      " << WARMUP << " runs  |  Timed: " << REPS
              << " runs\n\n";

    std::vector<double> results(NUM_INSTRUMENTS);

    // Benchmark helper: median of REPS timed runs after WARMUP warm-ups.
    auto bench = [&](const char* label, auto body) {
        std::vector<long long> timings;
        for (int r = 0; r < WARMUP + REPS; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            body();
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r >= WARMUP)
                timings.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                        .count());
        }
        std::sort(timings.begin(), timings.end());
        long long median = timings[timings.size() / 2];
        std::cout << label << median << " us\n";
        return median;
    };

    long long s = bench("Serial:             ", [&]() {
        for (int i = 0; i < NUM_INSTRUMENTS; i++)
            results[i] = heavy_compute(universe.price_series[i]);
    });

    // schedule(static): chunks assigned round-robin at fork time.
    long long t_static = bench("Parallel (static):  ", [&]() {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NUM_INSTRUMENTS; i++)
            results[i] = heavy_compute(universe.price_series[i]);
    });

    // schedule(dynamic, 4): threads pull batches of 4 from a shared queue.
    long long t_dynamic = bench("Parallel (dynamic): ", [&]() {
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < NUM_INSTRUMENTS; i++)
            results[i] = heavy_compute(universe.price_series[i]);
    });

    // schedule(guided): starts with large chunks, shrinks over time.
    long long t_guided = bench("Parallel (guided):  ", [&]() {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < NUM_INSTRUMENTS; i++)
            results[i] = heavy_compute(universe.price_series[i]);
    });

    std::cout << "\n--- Speedups vs Serial ---\n";
    std::cout << "Static:  " << static_cast<double>(s) / t_static << "x\n";
    std::cout << "Dynamic: " << static_cast<double>(s) / t_dynamic << "x\n";
    std::cout << "Guided:  " << static_cast<double>(s) / t_guided << "x\n";

    std::cout << "\nNote: With heterogeneous workloads, dynamic and guided\n"
              << "typically outperform static because they adapt to uneven\n"
              << "per-iteration costs. Profile your own workload to confirm.\n";

    return 0;
}
