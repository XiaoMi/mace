//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"
#include "mace/core/tensor.h"
#include "mace/kernels/addn.h"

using namespace mace;
using namespace mace::kernels;

static void AddNBenchmark(int iters, int n, int type) {
  const int64_t tot = static_cast<int64_t>(iters) * n * 3;
  mace::testing::ItemsProcessed(tot);
  mace::testing::BytesProcessed(tot * (sizeof(float)));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  vector<float> input1(n);
  vector<float> input2(n);
  vector<float> input3(n);
  vector<float> output(n);

  for (int64_t i = 0; i < n; ++i) {
    input1[i] = nd(gen);
    input2[i] = nd(gen);
    input3[i] = nd(gen);
  }
  vector<const float*> inputs { input1.data(), input2.data(), input3.data() };

  if (type == DeviceType::CPU) {
    AddNFunctor<DeviceType::CPU, float> addn_functor;
    addn_functor(inputs, &output[0], n);
  } else if (type == DeviceType::NEON) {
    AddNFunctor<DeviceType::NEON, float> neon_addn_functor;
    neon_addn_functor(inputs, &output[0], n);
  }
}

static const int kBenchmarkSize = 10000000;

BENCHMARK(AddNBenchmark)
    ->ArgPair(kBenchmarkSize, DeviceType::CPU)
    ->ArgPair(kBenchmarkSize, DeviceType::NEON);
