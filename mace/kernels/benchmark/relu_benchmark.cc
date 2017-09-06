//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"
#include "mace/core/tensor.h"
#include "mace/kernels/relu.h"

using namespace mace;
using namespace mace::kernels;

static void ReluBenchmark(int iters, int n, int type) {
  const int64_t tot = static_cast<int64_t>(iters) * n;
  mace::testing::ItemsProcessed(tot);
  mace::testing::BytesProcessed(tot * (sizeof(float)));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  vector<float> input(n);
  vector<float> output(n);

  for (int64_t i = 0; i < n; ++i) {
    input[i] = nd(gen);
  }

  if (type == DeviceType::CPU) {
    ReluFunctor<DeviceType::CPU, float> relu_functor;
    relu_functor(&input[0], &output[0], n);
  } else if (type == DeviceType::NEON) {
    ReluFunctor<DeviceType::NEON, float> neon_relu_functor;
    neon_relu_functor(&input[0], &output[0], n);
  }
}

static const int kBenchmarkSize = 10000000;

BENCHMARK(ReluBenchmark)
    ->ArgPair(kBenchmarkSize, DeviceType::CPU)
    ->ArgPair(kBenchmarkSize, DeviceType::NEON);
