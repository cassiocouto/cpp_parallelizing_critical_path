// Demonstrates the progression from data race -> critical section -> reduction.
// Shows why reduction is the right tool for aggregation in parallel loops.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "market_data.h"

int main() {
    const int NUM_INSTRUMENTS = 1000;
    const int TICKS = 1000;
    const double THRESHOLD = 0.5;
    const int TRIALS = 5;
    const int WARMUP = 3;
    const int REPS = 10;

    MarketUniverse universe;
    universe.generate(NUM_INSTRUMENTS, TICKS);

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "=== Race Conditions, Critical Sections & Reductions ===\n";
    std::cout << "Instruments: " << NUM_INSTRUMENTS
              << "  |  Threshold: " << THRESHOLD << "\n";
    std::cout << "Threads:     " << num_threads << "\n\n";

    // --- Serial baseline (ground truth) ---
    int serial_count = 0;
    for (int i = 0; i < NUM_INSTRUMENTS; i++) {
        double signal = compute_signal(universe.price_series[i]);
        if (std::abs(signal) > THRESHOLD) serial_count++;
    }
    std::cout << "Serial count (ground truth): " << serial_count << "\n\n";

    // --- Data race: run multiple times to show inconsistency ---
    std::cout << "1) DATA RACE (undefined behavior — count varies between runs):\n";
    for (int trial = 0; trial < TRIALS; trial++) {
        int signal_count = 0;
        #pragma omp parallel for
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            double signal = compute_signal(universe.price_series[i]);
            if (std::abs(signal) > THRESHOLD) {
                signal_count++;  // racy read-modify-write
            }
        }
        std::cout << "   Trial " << trial + 1 << ": " << signal_count
                  << (signal_count != serial_count ? "  <- WRONG" : "") << "\n";
    }

    // Benchmark helper: returns median over REPS timed runs after WARMUP
    auto bench = [&](auto body) {
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
        return timings[timings.size() / 2];
    };

    // --- Critical section: correct but serialized ---
    int critical_count = 0;
    long long critical_us = bench([&]() {
        critical_count = 0;
        #pragma omp parallel for
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            double signal = compute_signal(universe.price_series[i]);
            if (std::abs(signal) > THRESHOLD) {
                #pragma omp critical
                {
                    critical_count++;
                }
            }
        }
    });
    std::cout << "\n2) CRITICAL SECTION (correct, but serializes the increment):\n";
    std::cout << "   Count: " << critical_count << "  |  Median: " << critical_us
              << " us\n";

    // --- Atomic: lighter-weight synchronization ---
    int atomic_count = 0;
    long long atomic_us = bench([&]() {
        atomic_count = 0;
        #pragma omp parallel for
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            double signal = compute_signal(universe.price_series[i]);
            if (std::abs(signal) > THRESHOLD) {
                #pragma omp atomic
                atomic_count++;
            }
        }
    });
    std::cout << "\n3) ATOMIC (correct, lower overhead than critical):\n";
    std::cout << "   Count: " << atomic_count << "  |  Median: " << atomic_us
              << " us\n";

    // --- Reduction: correct and no contention ---
    int reduction_count = 0;
    long long reduction_us = bench([&]() {
        reduction_count = 0;
        #pragma omp parallel for reduction(+ : reduction_count)
        for (int i = 0; i < NUM_INSTRUMENTS; i++) {
            double signal = compute_signal(universe.price_series[i]);
            if (std::abs(signal) > THRESHOLD) {
                reduction_count++;
            }
        }
    });
    std::cout << "\n4) REDUCTION (correct, no locks, no contention):\n";
    std::cout << "   Count: " << reduction_count << "  |  Median: " << reduction_us
              << " us\n";

    std::cout << "\n--- Summary (median of " << REPS << " runs) ---\n";
    std::cout << "Critical:  " << critical_us << " us\n";
    std::cout << "Atomic:    " << atomic_us << " us\n";
    std::cout << "Reduction: " << reduction_us << " us  (winner)\n";

    return 0;
}
