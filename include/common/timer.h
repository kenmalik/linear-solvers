#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ENABLE_TIMER
inline constexpr bool timer_enabled = true;
#else
inline constexpr bool timer_enabled = false;
#endif

template <bool Enabled>
class CpuTimer;

template <>
class CpuTimer<true> {
  public:
    struct ScopedRange {
        ScopedRange(CpuTimer &timer, const char *name)
            : timer_(timer), name_(name) {
            timer_.start(name_);
        }

        ~ScopedRange() {
            timer_.stop(name_);
        }

        ScopedRange(const ScopedRange &) = delete;
        ScopedRange &operator=(const ScopedRange &) = delete;

      private:
        CpuTimer &timer_;
        const char *name_;
    };

    void start(const std::string &name) {
        register_name(name);
        starts_[name] = clock_t::now();
    }

    void stop(const std::string &name) {
        auto end = clock_t::now();
        auto ms = std::chrono::duration<double, std::milli>(
                      end - starts_.at(name))
                      .count();
        totals_[name] += ms;
        counts_[name]++;
    }

    void report(const std::string &fname) const {
        std::ofstream out{fname};
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

template <>
class CpuTimer<false> {
  public:
    struct ScopedRange {
        ScopedRange(CpuTimer &timer, const char *name) {}
        ~ScopedRange() {}
        ScopedRange(const ScopedRange &) = delete;
        ScopedRange &operator=(const ScopedRange &) = delete;
    };

    void start(const std::string &fname) {}
    void stop(const std::string &fname) {}
    void report(const std::string &fname) const {}
    void reset() {}
};

inline CpuTimer<timer_enabled> g_timer;
using CpuTimerRange = CpuTimer<timer_enabled>::ScopedRange;

static_assert(timer_enabled || sizeof(g_timer) == 1, "disabled CPU timer should have size of 1 byte");