#pragma once

#include "config.h"
#include "cuda_checks.h"

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

template <bool Enabled>
class CudaTimer;

// NOLINTBEGIN
template <>
class CudaTimer<false> {
  public:
    struct ScopedRange {
        ScopedRange(CudaTimer &, std::string_view, cudaStream_t) {}
    };

    void add_pair(std::string_view, cudaEvent_t, cudaEvent_t) {}
    void report(std::string_view) {}
};
// NOLINTEND

template <>
class CudaTimer<true> {
  public:
    struct ScopedRange {
        ScopedRange(CudaTimer &timer, std::string_view name, cudaStream_t stream)
            : timer_{timer}, name_{name}, stream_{stream} {
            timer_.register_name(name_);
            CUDA_CHECK(cudaEventCreate(&start_));
            CUDA_CHECK(cudaEventCreate(&stop_));
            CUDA_CHECK(cudaEventRecord(start_, stream_));
        }

        ScopedRange(CudaTimer &timer, const char *name, cudaStream_t stream)
            : ScopedRange{timer, std::string_view{name}, stream} {}

        ScopedRange(CudaTimer &timer, std::string &&name, cudaStream_t stream) = delete;

        ~ScopedRange() {
            CUDA_CHECK(cudaEventRecord(stop_, stream_));
            timer_.add_pair(name_, start_, stop_);
        }

        ScopedRange(const ScopedRange &) = delete;
        ScopedRange &operator=(const ScopedRange &) = delete;
        ScopedRange(ScopedRange &&) = delete;
        ScopedRange &operator=(ScopedRange &&) = delete;

      private:
        CudaTimer &timer_;
        std::string_view name_;
        cudaStream_t stream_;
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
    };

    void add_pair(std::string_view name, cudaEvent_t start, cudaEvent_t stop) {
        pairs_.push_back({.start = start, .stop = stop, .name = name});
    }

    void report(std::string_view fname) {
        CUDA_CHECK(cudaDeviceSynchronize());

        std::ostringstream oss;
        oss << fname;
        std::ofstream out{oss.str()};

        for (auto &p : pairs_) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, p.start, p.stop));
            totals_[p.name] += ms;
            counts_[p.name]++;
            CUDA_CHECK(cudaEventDestroy(p.start));
            CUDA_CHECK(cudaEventDestroy(p.stop));
        }
        out << "Range,Total (ms),Avg (ms),Instances\n";
        for (const auto &name : order_) {
            double total = totals_.at(name);
            int n = counts_.at(name);
            out << name << ',' << total << ',' << total / n << ',' << n
                << '\n';
        }
    }

    void reset() {
        for (auto &p : pairs_) {
            CUDA_CHECK(cudaEventDestroy(p.start));
            CUDA_CHECK(cudaEventDestroy(p.stop));
        }
        pairs_.clear();
        totals_.clear();
        counts_.clear();
        order_.clear();
        seen_.clear();
    }

  private:
    void register_name(std::string_view name) {
        if (seen_.insert(name).second) {
            order_.push_back(name);
        }
    }

    struct EventPair {
        cudaEvent_t start;
        cudaEvent_t stop;
        std::string_view name;
    };

    std::vector<EventPair> pairs_;
    std::unordered_map<std::string_view, double> totals_;
    std::unordered_map<std::string_view, int> counts_;
    std::vector<std::string_view> order_;
    std::unordered_set<std::string_view> seen_;
};

inline CudaTimer<timer_enabled> g_event_timer; // NOLINT
using CudaTimerRange = CudaTimer<timer_enabled>::ScopedRange;

static_assert(timer_enabled || sizeof(g_event_timer) == 1, "disabled CUDA timer should have size of 1 byte");
