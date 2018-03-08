//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

// Simple benchmarking facility.
#ifndef MACE_CORE_TESTING_TEST_BENCHMARK_H_
#define MACE_CORE_TESTING_TEST_BENCHMARK_H_

#include <string>
#include <utility>
#include <vector>

#define MACE_BENCHMARK_CONCAT(a, b, c) a##b##c
#define BENCHMARK(n)                                        \
  static ::mace::testing::Benchmark *MACE_BENCHMARK_CONCAT( \
      __benchmark_, n, __LINE__) = (new ::mace::testing::Benchmark(#n, (n)))

namespace mace {
namespace testing {

class Benchmark {
 public:
  Benchmark(const char *name, void (*benchmark_func)(int));

  static void Run();
  static void Run(const char *pattern);

 private:
  std::string name_;
  void (*benchmark_func_)(int iters) = nullptr;

  void Register();
  void Run(int *run_count, double *run_seconds);
};

void BytesProcessed(int64_t);
void MaccProcessed(int64_t);
void RestartTiming();
void StartTiming();
void StopTiming();

}  // namespace testing
}  // namespace mace

#endif  // MACE_CORE_TESTING_TEST_BENCHMARK_H_
