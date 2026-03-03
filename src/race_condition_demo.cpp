// race_condition_demo.cpp — Why naive shared-state updates break, and how
// to fix them.
//
// Counts how many instruments cross a signal threshold using four methods:
//
//   1. Data race        — unprotected signal_count++.  Runs multiple trials
//                         to show that the count varies non-deterministically.
//   2. Critical section — #pragma omp critical serializes the increment.
//                         Correct, but turns the hot path into a bottleneck.
//   3. Atomic           — #pragma omp atomic uses hardware atomics.
//                         Lower overhead than critical, still correct.
//   4. Reduction        — #pragma omp parallel for reduction(+:...) gives
//                         each thread a private copy and merges once at the
//                         end.  No locks, no contention — the right answer.
//
// Article sections: "Race Conditions and Critical Sections",
//                   "Reduction Clauses: The Right Tool"

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
    const int TRIALS = 5;    // repeated trials to expose the data race
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

    // -- Serial baseline: the ground truth we compare all methods against --
    int serial_count = 0;
    for (int i = 0; i < NUM_INSTRUMENTS; i++) {
        double signal = compute_signal(universe.price_series[i]);
        if (std::abs(signal) > THRESHOLD) serial_count++;
    }
    std::cout << "Serial count (ground truth): " << serial_count << "\n\n";

    // ------------------------------------------------------------------
    // 1) DATA RACE — undefined behavior.
    //    Multiple threads do signal_count++ on the same int without
    //    synchronization.  The read-modify-write is not atomic, so
    //    increments get lost.  Running several trials shows the count
    //    varies between runs — a classic symptom of a data race.
    // ------------------------------------------------------------------
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

    // Benchmark helper: median of REPS timed runs after WARMUP warm-up runs.
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

    // ------------------------------------------------------------------
    // 2) CRITICAL SECTION — correct but slow.
    //    Only one thread at a time can enter the critical block, which
    //    effectively serializes the increment.  On a hot path, the
    //    overhead comes from lock acquisition, thread serialization,
    //    and cache-line bouncing.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // 3) ATOMIC — correct, lighter than critical.
    //    Maps to a hardware compare-and-swap (or equivalent) for simple
    //    scalar updates.  Avoids the OS-level lock of critical, but
    //    threads still contend on the same cache line.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // 4) REDUCTION — correct, no contention.
    //    Each thread gets its own private copy of reduction_count.  All
    //    copies are combined with "+" exactly once after the parallel
    //    region ends.  No locks, no cache-line ping-pong — this is
    //    almost always the right tool for scalar aggregation.
    // ------------------------------------------------------------------
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
