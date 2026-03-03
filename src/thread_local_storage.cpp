// Demonstrates thread-private scratch buffers: variables declared inside a
// parallel region are automatically private per thread, avoiding false sharing
// and eliminating the need for locks.

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>

#include "market_data.h"

// Simulated windowed computation: for each non-overlapping window, fill a
// scratch buffer, compute log-returns, then a mini DFT (sum of harmonics).
// This is deliberately arithmetic-heavy to keep the workload CPU-bound.
double compute_windowed_metric(const std::vector<double>& prices,
                               std::vector<double>& scratch, int window) {
    int n = static_cast<int>(prices.size());
    if (n < window) return 0.0;

    double total = 0.0;
    int num_windows = (n - 1) / window;
    for (int w = 0; w < num_windows; w++) {
        int base = w * window;
        for (int j = 0; j < window; j++) {
            scratch[j] = std::log(prices[base + j + 1] / prices[base + j]);
        }

        // Mini DFT: compute energy of first 8 frequency bins
        const int NUM_BINS = 8;
        double energy = 0.0;
        for (int k = 0; k < NUM_BINS; k++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < window; j++) {
                double angle = 2.0 * std::numbers::pi * k * j / window;
                re += scratch[j] * std::cos(angle);
                im += scratch[j] * std::sin(angle);
            }
            energy += re * re + im * im;
        }
        total += std::sqrt(energy);
    }
    return total;
}

int main() {
    const int NUM_INSTRUMENTS = 500;
    const int TICKS = 2000;
    const int WINDOW_SIZE = 64;

    MarketUniverse universe;
    universe.generate(NUM_INSTRUMENTS, TICKS);

    std::vector<double> results(NUM_INSTRUMENTS);

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "=== Thread-Local Storage: Per-Thread Scratch Buffers ===\n";
    std::cout << "Instruments: " << NUM_INSTRUMENTS
              << "  |  Ticks: " << TICKS
              << "  |  Window: " << WINDOW_SIZE << "\n";
    std::cout << "Threads:     " << num_threads << "\n\n";

    // --- Serial ---
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        std::vector<double> scratch(WINDOW_SIZE);
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            results[i] = compute_windowed_metric(universe.price_series[i],
                                                  scratch, WINDOW_SIZE);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    long long serial_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // --- Parallel with thread-local scratch (declared inside parallel block) ---
    auto t2 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<double> local_buffer(WINDOW_SIZE);

        #pragma omp for
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            results[i] = compute_windowed_metric(universe.price_series[i],
                                                  local_buffer, WINDOW_SIZE);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    long long parallel_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cout << "Serial:   " << serial_us << " us\n";
    std::cout << "Parallel: " << parallel_us << " us\n";
    std::cout << "Speedup:  " << static_cast<double>(serial_us) / parallel_us
              << "x\n\n";

    // --- Demonstrate private clause (variable declared outside) ---
    double local_vol;
    std::vector<double> vol_result(NUM_INSTRUMENTS);

    auto t4 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for private(local_vol)
    for (int i = 0; i < NUM_INSTRUMENTS; i++) {
        local_vol = compute_volatility(universe.price_series[i]);
        vol_result[i] = local_vol;
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    long long private_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

    std::cout << "Private clause (volatility): " << private_us << " us\n";
    std::cout << "Sample — SYM0 max_window_vol=" << results[0]
              << "  overall_vol=" << vol_result[0] << "\n";

    return 0;
}
