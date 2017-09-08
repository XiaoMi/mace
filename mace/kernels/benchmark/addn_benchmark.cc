//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/testing/test_benchmark.h"
#include "mace/core/tensor.h"
#include "mace/kernels/addn.h"
#include "mace/kernels/neon/addn_neon.h"

using namespace mace;
using namespace mace::kernels;

static void AddNBenchmark(int iters, int n, int type) {
  const int64_t tot = static_cast<int64_t>(iters) * n * 3;
  mace::testing::ItemsProcessed(tot);
  mace::testing::BytesProcessed(tot * (sizeof(float)));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  Tensor input_tensor1(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor1.Resize({n});
  Tensor input_tensor2(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor2.Resize({n});
  Tensor input_tensor3(cpu_allocator(), DataType::DT_FLOAT);
  input_tensor3.Resize({n});
  vector<const Tensor*> input_tensors {&input_tensor1,
                                       &input_tensor2,
                                       &input_tensor3};
  Tensor output_tensor(cpu_allocator(), DataType::DT_FLOAT);
  output_tensor.ResizeLike(input_tensor1);
  float *input1 = input_tensor1.mutable_data<float>();
  float *input2 = input_tensor2.mutable_data<float>();
  float *input3 = input_tensor3.mutable_data<float>();
  float *output = output_tensor.mutable_data<float>();

  for (int64_t i = 0; i < n; ++i) {
    input1[i] = nd(gen);
    input2[i] = nd(gen);
    input3[i] = nd(gen);
  }

  if (type == DeviceType::CPU) {
    AddNFuntion<float>(input_tensors, &output_tensor);
  } else if (type == DeviceType::NEON) {
    NeonAddNFuntion_float(input_tensors, &output_tensor);
  }
}

static const int kBenchmarkSize = 10000000;

BENCHMARK(AddNBenchmark)
    ->ArgPair(kBenchmarkSize, DeviceType::CPU)
    ->ArgPair(kBenchmarkSize, DeviceType::NEON);
