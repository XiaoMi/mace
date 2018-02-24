//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <regex>
#include <vector>

#include "mace/core/testing/test_benchmark.h"
#include "mace/utils/env_time.h"
#include "mace/utils/logging.h"

namespace mace {
namespace testing {

static std::vector<Benchmark *> *all_benchmarks = nullptr;
static std::string label;
static int64_t bytes_processed;
static int64_t macc_processed;
static int64_t accum_time = 0;
static int64_t start_time = 0;

Benchmark::Benchmark(const char *name, void (*fn)(int))
    : name_(name), num_args_(0), fn0_(fn) {
  args_.push_back(std::make_pair(-1, -1));
  Register();
}

Benchmark::Benchmark(const char *name, void (*fn)(int, int))
    : name_(name), num_args_(1), fn1_(fn) {
  Register();
}

Benchmark::Benchmark(const char *name, void (*fn)(int, int, int))
    : name_(name), num_args_(2), fn2_(fn) {
  Register();
}

Benchmark *Benchmark::Arg(int x) {
  MACE_CHECK(num_args_ == 1);
  args_.push_back(std::make_pair(x, -1));
  return this;
}

Benchmark *Benchmark::ArgPair(int x, int y) {
  MACE_CHECK(num_args_ == 2);
  args_.push_back(std::make_pair(x, y));
  return this;
}

// Run all benchmarks
void Benchmark::Run() { Run("all"); }

void Benchmark::Run(const char *pattern) {
  if (!all_benchmarks) return;

  if (std::string(pattern) == "all") {
    pattern = ".*";
  }
  std::regex regex(pattern);

  // Compute name width.
  int width = 10;
  char name[100];
  std::smatch match;
  for (auto b : *all_benchmarks) {
    if (!std::regex_match(b->name_, match, regex)) continue;
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

  printf("%-*s %10s %10s %10s %10s\n", width, "Benchmark", "Time(ns)",
         "Iterations", "Input(MB/s)", "MACC(G/s)");
  printf("%s\n", std::string(width + 44, '-').c_str());
  for (auto b : *all_benchmarks) {
    if (!std::regex_match(b->name_, match, regex)) continue;
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

      float mbps = (bytes_processed * 1e-6) / seconds;
      // MACCs or other computations
      float gmaccs = (macc_processed * 1e-9) / seconds;
      printf("%-*s %10.0f %10d %10.2f %10.2f\n", width, name,
             seconds * 1e9 / iters, iters, mbps, gmaccs);
    }
  }
}

void Benchmark::Register() {
  if (!all_benchmarks) all_benchmarks = new std::vector<Benchmark *>;
  all_benchmarks->push_back(this);
}

void Benchmark::Run(int arg1, int arg2, int *run_count, double *run_seconds) {
  static const int64_t kMinIters = 10;
  static const int64_t kMaxIters = 1000000000;
  static const double kMinTime = 0.5;
  int64_t iters = kMinIters;
  while (true) {
    accum_time = 0;
    start_time = utils::NowMicros();
    bytes_processed = -1;
    macc_processed = -1;
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
void MaccProcessed(int64_t n) { macc_processed = n; }
void StartTiming() {
  if (start_time == 0) start_time = utils::NowMicros();
}
void StopTiming() {
  if (start_time != 0) {
    accum_time += (utils::NowMicros() - start_time);
    start_time = 0;
  }
}

}  // namespace testing
}  // namespace mace
