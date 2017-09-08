//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"
#include "mace/core/tensor.h"
#include "mace/kernels/relu.h"
#include "mace/kernels/neon/relu_neon.h"

using namespace mace;
using namespace mace::kernels;

static void ReluBenchmark(int iters, int n, int type) {
  const int64_t tot = static_cast<int64_t>(iters) * n;
  mace::testing::ItemsProcessed(tot);
  mace::testing::BytesProcessed(tot * (sizeof(float)));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  Tensor input_tensor(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor.Resize({n});
  Tensor output_tensor(cpu_allocator(), DataType::DT_FLOAT);
  output_tensor.ResizeLike(input_tensor);
  float *input = input_tensor.mutable_data<float>();
  float *output = output_tensor.mutable_data<float>();
  for (int64_t i = 0; i < n; ++i) {
    input[i] = nd(gen);
  }

  if (type == DeviceType::CPU) {
    ReluFuntion<float>(&input_tensor, &output_tensor);
  } else if (type == DeviceType::NEON) {
    NeonReluFuntion_float(&input_tensor, &output_tensor);
  }
}

static const int kBenchmarkSize = 10000000;

BENCHMARK(ReluBenchmark)
    ->ArgPair(kBenchmarkSize, DeviceType::CPU)
    ->ArgPair(kBenchmarkSize, DeviceType::NEON);
