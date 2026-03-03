// Demonstrates the basic parallel for pattern: computing EMA indicators
// independently across a universe of instruments.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iostream>

#include "market_data.h"

int main() {
    const int NUM_INSTRUMENTS = 500;
    const int TICKS = 1000;
    const int WARMUP = 3;
    const int REPS = 10;

    MarketUniverse universe;
    universe.generate(NUM_INSTRUMENTS, TICKS);

    std::vector<double> ema_fast(NUM_INSTRUMENTS);
    std::vector<double> ema_slow(NUM_INSTRUMENTS);

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "=== Parallel For: EMA Computation ===\n";
    std::cout << "Instruments: " << NUM_INSTRUMENTS
              << "  |  Ticks each: " << TICKS << "\n";
    std::cout << "Threads:     " << num_threads << "\n";
    std::cout << "Warmup:      " << WARMUP << " runs  |  Timed: " << REPS
              << " runs\n\n";

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

    long long serial_us = bench("Serial:   ", [&]() {
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            ema_fast[i] = compute_ema(universe.price_series[i], 12);
            ema_slow[i] = compute_ema(universe.price_series[i], 26);
        }
    });

    long long parallel_us = bench("Parallel: ", [&]() {
        #pragma omp parallel for
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            ema_fast[i] = compute_ema(universe.price_series[i], 12);
            ema_slow[i] = compute_ema(universe.price_series[i], 26);
        }
    });

    std::cout << "Speedup:  " << static_cast<double>(serial_us) / parallel_us
              << "x\n\n";

    // Full indicator update via Instrument method
    long long full_us = bench("Full indicator update (parallel): ", [&]() {
        #pragma omp parallel for
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            universe.instruments[i].update_indicators();
        }
    });

    std::cout << "\nSample — " << universe.instruments[0].symbol
              << ": fast_ema=" << universe.instruments[0].fast_ema
              << "  slow_ema=" << universe.instruments[0].slow_ema
              << "  vol=" << universe.instruments[0].volatility << "\n";

    (void)full_us;
    return 0;
}
