// market_data.h — Shared types and helpers for every example in this project.
//
// Provides:
//   Instrument       — a single tradeable symbol with its price history and
//                      cached indicator values (fast/slow EMA, volatility).
//   compute_ema()    — exponential moving average (free-function form).
//   compute_volatility() — population standard deviation of a price series.
//   compute_signal() — MACD-style signal (fast EMA − slow EMA).
//   MarketUniverse   — generates a universe of synthetic instruments, either
//                      homogeneous (uniform tick counts) or heterogeneous
//                      (variable tick counts, mimicking mixed-liquidity names).

#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Instrument — one symbol in the universe.
//
// Holds its own price series and cached indicator outputs.  The member
// methods operate on the stored series; the free functions below accept
// any price vector and are used by examples that work with raw data.
// ---------------------------------------------------------------------------
struct Instrument {
    std::string symbol;
    std::vector<double> price_series;
    double fast_ema = 0.0;
    double slow_ema = 0.0;
    double volatility = 0.0;

    // Recompute all cached indicators — safe to call from a parallel region
    // because each Instrument writes only to its own members.
    void update_indicators() {
        fast_ema = compute_ema(12);
        slow_ema = compute_ema(26);
        volatility = compute_volatility();
    }

    // EMA: alpha = 2/(period+1), single pass.  Standard recurrence:
    //   ema(t) = alpha * price(t) + (1 - alpha) * ema(t-1)
    double compute_ema(int period) const {
        if (price_series.empty()) return 0.0;
        double alpha = 2.0 / (period + 1);
        double ema = price_series[0];
        for (size_t t = 1; t < price_series.size(); t++) {
            ema = alpha * price_series[t] + (1.0 - alpha) * ema;
        }
        return ema;
    }

    // Population standard deviation of the price series.
    double compute_volatility() const {
        if (price_series.empty()) return 0.0;
        double mean = 0.0;
        for (double p : price_series) mean += p;
        mean /= static_cast<double>(price_series.size());

        double variance = 0.0;
        for (double p : price_series) {
            double diff = p - mean;
            variance += diff * diff;
        }
        return std::sqrt(variance / static_cast<double>(price_series.size()));
    }
};

// ---------------------------------------------------------------------------
// Free-function variants — take any price vector, used by examples that
// operate on MarketUniverse::price_series directly.
// ---------------------------------------------------------------------------

inline double compute_ema(const std::vector<double>& prices, int period) {
    if (prices.empty()) return 0.0;
    double alpha = 2.0 / (period + 1);
    double ema = prices[0];
    for (size_t t = 1; t < prices.size(); t++) {
        ema = alpha * prices[t] + (1.0 - alpha) * ema;
    }
    return ema;
}

inline double compute_volatility(const std::vector<double>& prices) {
    if (prices.empty()) return 0.0;
    double mean = 0.0;
    for (double p : prices) mean += p;
    mean /= static_cast<double>(prices.size());

    double variance = 0.0;
    for (double p : prices) {
        double diff = p - mean;
        variance += diff * diff;
    }
    return std::sqrt(variance / static_cast<double>(prices.size()));
}

// MACD-style crossover signal: fast EMA(12) minus slow EMA(26).
// Positive → bullish momentum, negative → bearish.
inline double compute_signal(const std::vector<double>& prices) {
    double fast = compute_ema(prices, 12);
    double slow = compute_ema(prices, 26);
    return fast - slow;
}

// ---------------------------------------------------------------------------
// MarketUniverse — synthetic data generator.
//
// Two generation modes:
//   generate()              — every instrument gets the same number of ticks.
//   generate_heterogeneous()— tick counts drawn from [200, 5000], simulating
//                             a mix of liquid large-caps and illiquid names.
//
// Price series are geometric random walks: P(t) = P(t-1) * (1 + noise),
// where noise ~ N(0, 0.01).  Base prices are uniform in [20, 500].
// ---------------------------------------------------------------------------
struct MarketUniverse {
    std::vector<Instrument> instruments;
    std::vector<std::vector<double>> price_series;

    void generate(int num_instruments, int ticks_per_instrument, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> noise(0.0, 0.01);
        std::uniform_real_distribution<double> base_price(20.0, 500.0);

        instruments.resize(num_instruments);
        price_series.resize(num_instruments);

        for (int i = 0; i < num_instruments; i++) {
            price_series[i].resize(ticks_per_instrument);
            price_series[i][0] = base_price(rng);
            for (int t = 1; t < ticks_per_instrument; t++) {
                price_series[i][t] = price_series[i][t - 1] * (1.0 + noise(rng));
            }

            instruments[i].symbol = "SYM" + std::to_string(i);
            instruments[i].price_series = price_series[i];
        }
    }

    void generate_heterogeneous(int num_instruments, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> noise(0.0, 0.01);
        std::uniform_real_distribution<double> base_price(20.0, 500.0);
        std::uniform_int_distribution<int> tick_count(200, 5000);

        instruments.resize(num_instruments);
        price_series.resize(num_instruments);

        for (int i = 0; i < num_instruments; i++) {
            int ticks = tick_count(rng);
            price_series[i].resize(ticks);
            price_series[i][0] = base_price(rng);
            for (int t = 1; t < ticks; t++) {
                price_series[i][t] = price_series[i][t - 1] * (1.0 + noise(rng));
            }

            instruments[i].symbol = "SYM" + std::to_string(i);
            instruments[i].price_series = price_series[i];
        }
    }
};
