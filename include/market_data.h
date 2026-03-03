#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

struct Instrument {
    std::string symbol;
    std::vector<double> price_series;
    double fast_ema = 0.0;
    double slow_ema = 0.0;
    double volatility = 0.0;

    void update_indicators() {
        fast_ema = compute_ema(12);
        slow_ema = compute_ema(26);
        volatility = compute_volatility();
    }

    double compute_ema(int period) const {
        if (price_series.empty()) return 0.0;
        double alpha = 2.0 / (period + 1);
        double ema = price_series[0];
        for (size_t t = 1; t < price_series.size(); t++) {
            ema = alpha * price_series[t] + (1.0 - alpha) * ema;
        }
        return ema;
    }

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

inline double compute_signal(const std::vector<double>& prices) {
    double fast = compute_ema(prices, 12);
    double slow = compute_ema(prices, 26);
    return fast - slow;
}

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

    // Heterogeneous universe: liquid names get more ticks, illiquid get fewer.
    // Useful for demonstrating scheduling strategy differences.
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
