#pragma once

#include "cuda_checks.h"
#include "timer.h"

#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <bool Enabled> class CudaEventTimer {
  public:
    struct ScopedRange {
        ScopedRange(CudaEventTimer &timer, const char *name,
                    cudaStream_t stream)
            : timer_(timer), name_(name), stream_(stream) {
            if constexpr (Enabled) {
                timer_.register_name(name_);
                CUDA_CHECK(cudaEventCreate(&start_));
                CUDA_CHECK(cudaEventCreate(&stop_));
                CUDA_CHECK(cudaEventRecord(start_, stream_));
            }
        }

        ~ScopedRange() {
            if constexpr (Enabled) {
                CUDA_CHECK(cudaEventRecord(stop_, stream_));
                timer_.add_pair(name_, start_, stop_);
            }
        }

        ScopedRange(const ScopedRange &) = delete;
        ScopedRange &operator=(const ScopedRange &) = delete;

      private:
        CudaEventTimer &timer_;
        const char *name_;
        cudaStream_t stream_;
        cudaEvent_t start_;
        cudaEvent_t stop_;
    };

    void add_pair(const char *name, cudaEvent_t start, cudaEvent_t stop) {
        if constexpr (Enabled) {
            pairs_.push_back({start, stop, std::string(name)});
        }
    }

    void report(const std::string &fname) {
        if constexpr (Enabled) {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::ofstream out{fname};

            for (auto &p : pairs_) {
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, p.start, p.stop));
                totals_[p.name] += ms;
                counts_[p.name]++;
                CUDA_CHECK(cudaEventDestroy(p.start));
                CUDA_CHECK(cudaEventDestroy(p.stop));
            }
            pairs_.clear();
            out << "Range,Total (ms),Avg (ms),Instances\n";
            for (const auto &name : order_) {
                double total = totals_.at(name);
                int n = counts_.at(name);
                out << name << ',' << total << ',' << total / n << ',' << n
                    << '\n';
            }
        }
    }

    void reset() {
        if constexpr (Enabled) {
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
    }

  private:
    void register_name(const std::string &name) {
        if (seen_.insert(name).second) {
            order_.push_back(name);
        }
    }

    struct EventPair {
        cudaEvent_t start;
        cudaEvent_t stop;
        std::string name;
    };

    std::vector<EventPair> pairs_;
    std::unordered_map<std::string, double> totals_;
    std::unordered_map<std::string, int> counts_;
    std::vector<std::string> order_;
    std::unordered_set<std::string> seen_;
};

inline CudaEventTimer<timer_enabled> g_event_timer;
using CudaEventRange = CudaEventTimer<timer_enabled>::ScopedRange;
