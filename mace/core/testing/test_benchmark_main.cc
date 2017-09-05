//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <iostream>

#include "mace/core/testing/test_benchmark.h"

int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc\n";

  mace::testing::Benchmark::Run();
  return 0;
}

