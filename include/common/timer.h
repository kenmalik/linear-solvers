#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_TIMER
inline constexpr bool timer_enabled = true;
#else
inline constexpr bool timer_enabled = false;
#endif

template <bool Enabled> class SectionTimer {
  public:
    void start(const std::string &name) {
        if constexpr (Enabled) {
            register_name(name);
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

    void report(const std::string &fname) const {
        if constexpr (Enabled) {
            std::ofstream out{fname};
            out << "Range,Total (ms),Avg (ms),Instances\n";
            for (const auto &name : order_) {
                double total = totals_.at(name);
                int n = counts_.at(name);
                out << name << ',' << total << "," << total / n << ',' << n
                    << '\n';
            }
        }
    }

    void reset() {
        if constexpr (Enabled) {
            starts_.clear();
            totals_.clear();
            counts_.clear();
            order_.clear();
            seen_.clear();
        }
    }

  private:
    using clock_t = std::chrono::steady_clock;

    void register_name(const std::string &name) {
        if (seen_.insert(name).second) {
            order_.push_back(name);
        }
    }

    std::unordered_map<std::string, clock_t::time_point> starts_;
    std::unordered_map<std::string, double> totals_;
    std::unordered_map<std::string, int> counts_;
    std::vector<std::string> order_;
    std::unordered_set<std::string> seen_;
};

inline SectionTimer<timer_enabled> g_timer;
