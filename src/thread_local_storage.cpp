// thread_local_storage.cpp — Per-thread scratch buffers and the private clause.
//
// Two patterns for giving each thread its own working memory:
//
//   Pattern A — declare inside the parallel region.
//       Variables declared inside a #pragma omp parallel block are
//       automatically private to each thread.  Each thread gets its own
//       stack-allocated instance, so there is zero contention.
//
//   Pattern B — private() clause.
//       When the variable is declared *outside* the parallel region
//       (e.g. a loop-scoped temp), the private(var) clause tells
//       OpenMP to create a thread-local copy.  The original variable
//       is left untouched; each thread works on its own copy.
//
// Both patterns avoid false sharing (threads writing to adjacent memory
// on the same cache line) and eliminate the need for any locks.
//
// Article section: "Thread-Local Storage"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>

#include "market_data.h"

// ---------------------------------------------------------------------------
// Windowed spectral metric.
//
// For each non-overlapping window:
//   1. Compute log-returns into the scratch buffer.
//   2. Run a partial DFT (8 bins) and accumulate spectral energy.
//
// The scratch buffer is mutated on every window, which is why it must be
// private per thread — two threads sharing the same buffer would corrupt
// each other's intermediate results.
// ---------------------------------------------------------------------------
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

    // -- Serial baseline: one scratch buffer reused across all instruments --
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

    // -- Pattern A: scratch buffer declared inside the parallel region ------
    // Each thread creates its own local_buffer on its own stack.
    // No synchronization needed — it's private by construction.
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

    // -- Pattern B: private clause on a variable declared outside -----------
    // local_vol is declared here (outer scope), but private(local_vol)
    // gives each thread its own uninitialized copy.  Writes to local_vol
    // inside the loop don't interfere across threads.
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
