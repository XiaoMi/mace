//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

// Simple benchmarking facility.
#ifndef MACE_TEST_BENCHMARK_H_
#define MACE_TEST_BENCHMARK_H_

#include <utility>
#include <vector>

#include "mace/core/types.h"

#define MACE_BENCHMARK_CONCAT(a, b, c) a##b##c
#define BENCHMARK(n)                                            \
  static ::mace::testing::Benchmark* MACE_BENCHMARK_CONCAT(__benchmark_, n, __LINE__) = \
      (new ::mace::testing::Benchmark(#n, (n)))

namespace mace {
namespace testing {

class Benchmark {
 public:
  Benchmark(const char* name, void (*fn)(int));
  Benchmark(const char* name, void (*fn)(int, int));
  Benchmark(const char* name, void (*fn)(int, int, int));
  Benchmark* Arg(int x);
  Benchmark* ArgPair(int x, int y);

  static void Run();
  static void Run(const char* pattern);

 private:
  string name_;
  int num_args_;
  std::vector<std::pair<int, int>> args_;
  void (*fn0_)(int) = nullptr;
  void (*fn1_)(int, int) = nullptr;
  void (*fn2_)(int, int, int) = nullptr;

  void Register();
  void Run(int arg1, int arg2, int* run_count, double* run_seconds);
};

void RunBenchmarks();
void BytesProcessed(int64_t);
void ItemsProcessed(int64_t);
void StartTiming();
void StopTiming();

}  // namespace testing
}  // namespace mace

#endif  // MACE_TEST_BENCHMARK_H_
