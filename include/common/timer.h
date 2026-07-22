#pragma once

#include "config.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <bool Enabled>
class CpuTimer;

// NOLINTBEGIN
template <>
class CpuTimer<false> {
  public:
    struct ScopedRange {
        ScopedRange(CpuTimer &, std::string_view) {}
    };

    void start(std::string_view) {}
    void stop(std::string_view) {}
    void report(std::string_view) const {}
};
// NOLINTEND

template <>
class CpuTimer<true> {
  public:
    struct ScopedRange {
        ScopedRange(CpuTimer &timer, std::string_view name)
            : timer_{timer}, name_{name} {
            timer_.start(name_);
        }

        ScopedRange(CpuTimer &timer, const char *name)
            : ScopedRange{timer, std::string_view{name}} {}

        ScopedRange(CpuTimer &timer, std::string &&name) = delete;

        ScopedRange(const ScopedRange &) = delete;
        ScopedRange &operator=(const ScopedRange &) = delete;
        ScopedRange(ScopedRange &&) = delete;
        ScopedRange &operator=(ScopedRange &&) = delete;

        ~ScopedRange() {
            timer_.stop(name_);
        }

      private:
        CpuTimer &timer_;
        std::string_view name_;
    };

    void start(std::string_view name) {
        register_name(name);
        starts_[name] = clock_t::now();
    }

    void stop(std::string_view name) {
        auto end = clock_t::now();
        auto ms = std::chrono::duration<double, std::milli>(
                      end - starts_.at(name))
                      .count();
        totals_[name] += ms;
        counts_[name]++;
    }

    void report(std::string_view fname) const {
        std::ostringstream oss;
        oss << fname;
        std::ofstream out{oss.str()};
        out << "Range,Total (ms),Avg (ms),Instances\n";
        for (const auto &name : order_) {
            double total = totals_.at(name);
            int n = counts_.at(name);
            out << name << ',' << total << "," << total / n << ',' << n
                << '\n';
        }
    }

    void reset() {
        starts_.clear();
        totals_.clear();
        counts_.clear();
        order_.clear();
        seen_.clear();
    }

  private:
    using clock_t = std::chrono::steady_clock;

    void register_name(std::string_view name) {
        if (seen_.insert(name).second) {
            order_.push_back(name);
        }
    }

    std::unordered_map<std::string_view, clock_t::time_point> starts_;
    std::unordered_map<std::string_view, double> totals_;
    std::unordered_map<std::string_view, int> counts_;
    std::vector<std::string_view> order_;
    std::unordered_set<std::string_view> seen_;
};

inline CpuTimer<timer_enabled> g_timer; // NOLINT
using CpuTimerRange = CpuTimer<timer_enabled>::ScopedRange;

static_assert(timer_enabled || sizeof(g_timer) == 1, "disabled CPU timer should have size of 1 byte");
