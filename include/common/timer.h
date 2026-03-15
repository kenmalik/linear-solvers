#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

#ifdef ENABLE_TIMER
inline constexpr bool timer_enabled = true;
#else
inline constexpr bool timer_enabled = false;
#endif

template <bool Enabled> class SectionTimer {
  public:
    void start(const std::string &name) {
        if constexpr (Enabled) {
            starts_[name] = clock_t::now();
        }
    }

    void stop(const std::string &name) {
        if constexpr (Enabled) {
            auto end = clock_t::now();
            auto ms = std::chrono::duration<double, std::milli>(
                          end - starts_.at(name))
                          .count();
            totals_[name] += ms;
            counts_[name]++;
        }
    }

    void report(std::ostream &out = std::cout) const {
        if constexpr (Enabled) {
            out << "\n--- Timing Report ---\n";
            for (const auto &[name, total] : totals_) {
                int n = counts_.at(name);
                out << name << ": " << total << " ms total, " << total / n
                    << " ms avg (" << n << " calls)\n";
            }
            out << "---------------------\n";
        }
    }

    void reset() {
        if constexpr (Enabled) {
            starts_.clear();
            totals_.clear();
            counts_.clear();
        }
    }

  private:
    using clock_t = std::chrono::steady_clock;

    std::unordered_map<std::string, clock_t::time_point> starts_;
    std::unordered_map<std::string, double> totals_;
    std::unordered_map<std::string, int> counts_;
};

inline SectionTimer<timer_enabled> g_timer;
