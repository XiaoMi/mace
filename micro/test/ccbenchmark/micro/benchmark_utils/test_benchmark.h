// Copyright 2019 The MICRO Authors. All Rights Reserved.
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

// Simple benchmarking facility.
#ifndef MICRO_TEST_CCBENCHMARK_MICRO_BENCHMARK_UTILS_TEST_BENCHMARK_H_
#define MICRO_TEST_CCBENCHMARK_MICRO_BENCHMARK_UTILS_TEST_BENCHMARK_H_

#include <stdlib.h>

#define MICRO_BENCHMARK(n) \
  static ::micro::testing::Benchmark __benchmark_##n(#n, (n))

namespace micro {
namespace testing {

typedef void BenchmarkFunc(int32_t iters);

class Benchmark {
 public:
  Benchmark(const char *name, BenchmarkFunc *benchmark_func);

  static void Run();

 private:
  const char *name_;
  BenchmarkFunc *benchmark_func_;

  void Register();
  void Run(int32_t *run_count, double *run_seconds);
};

void BytesProcessed(int64_t);
void MacsProcessed(int64_t);
void RestartTiming();
void StartTiming();
void StopTiming();

}  // namespace testing
}  // namespace micro

extern "C" {
void BenchmarkRun();
}

#endif  // MICRO_TEST_CCBENCHMARK_MICRO_BENCHMARK_UTILS_TEST_BENCHMARK_H_
