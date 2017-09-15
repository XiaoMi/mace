//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <iostream>

#include "mace/core/testing/test_benchmark.h"

int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc\n";

  // TODO Use gflags
  if (argc == 2) {
    mace::testing::Benchmark::Run(argv[1]);
  } else {
    mace::testing::Benchmark::Run("all");
  }
  return 0;
}
