//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <vector>
#include "mace/core/logging.h"
#include "mace/core/testing/env_time.h"
#include "mace/core/testing/test_benchmark.h"

namespace mace {
namespace testing {

static std::vector<Benchmark*>* all_benchmarks = nullptr;
static std::string label;
static int64_t bytes_processed;
static int64_t items_processed;
static int64_t accum_time = 0;
static int64_t start_time = 0;

Benchmark::Benchmark(const char* name, void (*fn)(int))
    : name_(name), num_args_(0), fn0_(fn) {
  args_.push_back(std::make_pair(-1, -1));
  Register();
}

Benchmark::Benchmark(const char* name, void (*fn)(int, int))
    : name_(name), num_args_(1), fn1_(fn) {
  Register();
}

Benchmark::Benchmark(const char* name, void (*fn)(int, int, int))
    : name_(name), num_args_(2), fn2_(fn) {
  Register();
}

Benchmark* Benchmark::Arg(int x) {
  MACE_CHECK(num_args_ == 1);
  args_.push_back(std::make_pair(x, -1));
  return this;
}

Benchmark* Benchmark::ArgPair(int x, int y) {
  MACE_CHECK(num_args_ == 2);
  args_.push_back(std::make_pair(x, y));
  return this;
}


// Run all benchmarks
void Benchmark::Run() {
  if (!all_benchmarks) return;

  // Compute name width.
  int width = 10;
  char name[100];
  for (auto b : *all_benchmarks) {
    for (auto arg : b->args_) {
      strcpy(name, b->name_.c_str());
      if (arg.first >= 0) {
        sprintf(name, "%s/%d", name, arg.first);
        if (arg.second >= 0) {
          sprintf(name, "%s/%d", name, arg.second);
        }
      }

      width = std::max<int>(width, strlen(name));
    }
  }

  printf("%-*s %10s %10s\n", width, "Benchmark", "Time(ns)", "Iterations");
  printf("%s\n", string(width + 22, '-').c_str());
  for (auto b : *all_benchmarks) {

    for (auto arg : b->args_) {
      strcpy(name, b->name_.c_str());
      if (arg.first >= 0) {
        sprintf(name, "%s/%d", name, arg.first);
        if (arg.second >= 0) {
          sprintf(name, "%s/%d", name, arg.second);
        }
      }

      int iters;
      double seconds;
      b->Run(arg.first, arg.second, &iters, &seconds);

      char buf[100];
      std::string full_label = label;
      if (bytes_processed > 0) {
        snprintf(buf, sizeof(buf), " %.1fMB/s",
                 (bytes_processed * 1e-6) / seconds);
        full_label += buf;
      }
      if (items_processed > 0) {
        snprintf(buf, sizeof(buf), " %.1fM items/s",
                 (items_processed * 1e-6) / seconds);
        full_label += buf;
      }
      printf("%-*s %10.0f %10d\t%s\n", width, name,
             seconds * 1e9 / iters, iters, full_label.c_str());
    }
  }
}

void Benchmark::Register() {
  if (!all_benchmarks) all_benchmarks = new std::vector<Benchmark*>;
  all_benchmarks->push_back(this);
}

void Benchmark::Run(int arg1, int arg2, int* run_count, double* run_seconds) {
  static const int64_t kMinIters = 100;
  static const int64_t kMaxIters = 1000000000;
  static const double kMinTime = 0.5;
  int64_t iters = kMinIters;
  while (true) {
    accum_time = 0;
    start_time = NowMicros();
    bytes_processed = -1;
    items_processed = -1;
    label.clear();
    if (fn0_) {
      (*fn0_)(iters);
    } else if (fn1_) {
      (*fn1_)(iters, arg1);
    } else {
      (*fn2_)(iters, arg1, arg2);
    }
    StopTiming();
    const double seconds = accum_time * 1e-6;
    if (seconds >= kMinTime || iters >= kMaxIters) {
      *run_count = iters;
      *run_seconds = seconds;
      return;
    }

    // Update number of iterations.  Overshoot by 40% in an attempt
    // to succeed the next time.
    double multiplier = 1.4 * kMinTime / std::max(seconds, 1e-9);
    multiplier = std::min(10.0, multiplier);
    if (multiplier <= 1.0) multiplier *= 2.0;
    iters = std::max<int64_t>(multiplier * iters, iters + 1);
    iters = std::min(iters, kMaxIters);
  }
}

void BytesProcessed(int64_t n) { bytes_processed = n; }
void ItemsProcessed(int64_t n) { items_processed = n; }
void StartTiming() {
  if (start_time == 0) start_time = NowMicros();
}
void StopTiming() {
  if (start_time != 0) {
    accum_time += (NowMicros() - start_time);
    start_time = 0;
  }
}

}  // namespace testing
}  // namespace mace
