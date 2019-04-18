// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <regex>  // NOLINT(build/c++11)
#include <vector>

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/port/env.h"
#include "mace/utils/logging.h"

namespace mace {
namespace testing {

static std::vector<Benchmark *> *all_benchmarks = nullptr;
static int64_t bytes_processed;
static int64_t macs_processed = 0;
static int64_t accum_time = 0;
static int64_t start_time = 0;

Benchmark::Benchmark(const char *name, void (*benchmark_func)(int))
    : name_(name), benchmark_func_(benchmark_func) {
  Register();
}

// Run all benchmarks that matches the pattern
void Benchmark::Run(const char *pattern) {
  if (!all_benchmarks) return;

  std::sort(all_benchmarks->begin(), all_benchmarks->end(),
            [](const Benchmark *lhs, const Benchmark *rhs) {
              return lhs->name_ < rhs->name_;
            });

  if (std::string(pattern) == "all") {
    pattern = ".*";
  }
  std::regex regex(pattern);

  // Compute name width.
  int width = 10;
  std::smatch match;
  for (auto b : *all_benchmarks) {
    if (!std::regex_match(b->name_, match, regex)) continue;
    width = std::max<int>(width, b->name_.length());
  }

  // Internal perf regression tools depends on the output formatting,
  // please keep in consistent when modifying
  printf("%-*s %10s %10s %10s %10s\n", width, "Benchmark", "Time(ns)",
         "Iterations", "Input(MB/s)", "GMACPS");
  printf("%s\n", std::string(width + 45, '-').c_str());
  for (auto b : *all_benchmarks) {
    if (!std::regex_match(b->name_, match, regex)) continue;
    int iters;
    double seconds;
    b->Run(&iters, &seconds);
    float mbps = (bytes_processed * 1e-6) / seconds;
    // MACCs or other computations
    float gmacs = (macs_processed * 1e-9) / seconds;
    printf("%-*s %10.0f %10d %10.2f %10.2f\n", width, b->name_.c_str(),
           seconds * 1e9 / iters, iters, mbps, gmacs);
  }
}

void Benchmark::Register() {
  if (!all_benchmarks) all_benchmarks = new std::vector<Benchmark *>;
  all_benchmarks->push_back(this);
}

void Benchmark::Run(int *run_count, double *run_seconds) {
  static const int64_t kMinIters = 10;
  static const int64_t kMaxIters = 1000000000;
  static const double kMinTime = 0.5;
  int64_t iters = kMinIters;
  while (true) {
    bytes_processed = -1;
    macs_processed = 0;
    RestartTiming();
    (*benchmark_func_)(iters);
    StopTiming();
    const double seconds = accum_time * 1e-6;
    if (seconds >= kMinTime || iters >= kMaxIters) {
      *run_count = iters;
      *run_seconds = seconds;
      return;
    }

    // Update number of iterations.
    // Overshoot by 100% in an attempt to succeed the next time.
    double multiplier = 2.0 * kMinTime / std::max(seconds, 1e-9);
    iters = std::min<int64_t>(multiplier * iters, kMaxIters);
  }
}

void BytesProcessed(int64_t n) { bytes_processed = n; }
void MacsProcessed(int64_t n) { macs_processed = n; }
void RestartTiming() {
  accum_time = 0;
  start_time = NowMicros();
}
void StartTiming() { start_time = NowMicros(); }
void StopTiming() {
  if (start_time != 0) {
    accum_time += (NowMicros() - start_time);
    start_time = 0;
  }
}

}  // namespace testing
}  // namespace mace
