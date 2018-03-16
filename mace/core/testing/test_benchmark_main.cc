//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <iostream>

#include "mace/core/testing/test_benchmark.h"
#include "mace/public/mace.h"

int main(int argc, char **argv) {
  std::cout << "Running main() from test_main.cc\n";

  mace::ConfigCPURuntime(4, mace::CPUPowerOption::HIGH_PERFORMANCE);
  mace::ConfigOpenCLRuntime(mace::GPUType::ADRENO, mace::GPUPerfHint::PERF_HIGH,
                            mace::GPUPriorityHint::PRIORITY_HIGH);

  if (argc == 2) {
    mace::testing::Benchmark::Run(argv[1]);
  } else {
    mace::testing::Benchmark::Run("all");
  }
  return 0;
}
