// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template<DeviceType D, typename T>
void BMSplitHelper(int iters,
                   const std::vector<index_t> &input_shape,
                   const index_t num_outputs) {
  mace::testing::StopTiming();

  // Construct graph
  OpsTestNet net;

  const index_t input_size = std::accumulate(input_shape.begin(),
                                             input_shape.end(),
                                             1,
                                             std::multiplies<index_t>());
  std::vector<float> input_data(input_size);
  GenerateRandomRealTypeData(input_shape, &input_data);
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);

  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);

    auto builder = OpDefBuilder("Split", "SplitTest");
    builder.Input("InputImage");
    for (int i = 0; i < num_outputs; ++i) {
      builder = builder.Output(MakeString("OutputImage", i));
    }
    builder
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    auto builder = OpDefBuilder("Split", "SplitTest");
    builder.Input("Input");
    for (int i = 0; i < num_outputs; ++i) {
      builder = builder.Output(MakeString("Output", i));
    }
    builder.Finalize(net.NewOperatorDef());
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.RunOp(D);
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_SPLIT_MACRO(N, H, W, C, NO, TYPE, DEVICE)                    \
  static void                                                                \
      MACE_BM_SPLIT_##N##_##H##_##W##_##C##_##NO##_##TYPE##_##DEVICE(        \
          int iters) {                                                       \
        const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;     \
        mace::testing::MaccProcessed(tot);                                   \
        mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
        BMSplitHelper<DEVICE, TYPE>(iters, {N, H, W, C}, NO);                \
      }                                                                      \
      MACE_BENCHMARK(                                                        \
          MACE_BM_SPLIT_##N##_##H##_##W##_##C##_##NO##_##TYPE##_##DEVICE)

#define MACE_BM_SPLIT(N, H, W, C, NO)                 \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, float, CPU);    \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, float, GPU);    \
  MACE_BM_SPLIT_MACRO(N, H, W, C, NO, half, GPU);

MACE_BM_SPLIT(1, 32, 32, 32, 2);
MACE_BM_SPLIT(1, 32, 32, 128, 2);
MACE_BM_SPLIT(1, 32, 32, 256, 2);
MACE_BM_SPLIT(1, 128, 128, 32, 2);
MACE_BM_SPLIT(1, 128, 128, 128, 2);

}  // namespace test
}  // namespace ops
}  // namespace mace
