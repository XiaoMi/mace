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

#include <string>

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/kernels/eltwise.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void EltwiseBenchmark(
    int iters, kernels::EltwiseType type, int n, int h, int w, int c) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, T>("Input0", {n, h, w, c});
  net.AddRandomInput<D, T>("Input1", {n, h, w, c});

  if (D == DeviceType::GPU) {
    BufferToImage<D, half>(&net, "Input0", "InputImg0",
                           kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, half>(&net, "Input1", "InputImg1",
                           kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("InputImg0")
        .Input("InputImg1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", {1.2, 2.1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Output("OutputImg")
        .Finalize(net.NewOperatorDef());
  } else {
    net.TransformDataFormat<D, float>("Input0", NHWC,
                                      "TInput0", NCHW);
    net.TransformDataFormat<D, float>("Input1", NHWC,
                                      "TInput1", NCHW);
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("TInput0")
        .Input("TInput1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", {1.2, 2.1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
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

#define MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, TYPE, DEVICE)             \
  static void                                                                 \
      MACE_BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE( \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;          \
    mace::testing::MaccProcessed(tot);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    EltwiseBenchmark<DEVICE, TYPE>(                                           \
        iters, static_cast<kernels::EltwiseType>(ELT_TYPE), N, H, W, C);      \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define MACE_BM_ELTWISE(ELT_TYPE, N, H, W, C)                 \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, CPU);    \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, GPU);    \
  MACE_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, half, GPU);

MACE_BM_ELTWISE(2, 1, 128, 128, 32);
MACE_BM_ELTWISE(2, 1, 240, 240, 256);
MACE_BM_ELTWISE(2, 1, 256, 256, 32);
MACE_BM_ELTWISE(0, 1, 128, 128, 32);
MACE_BM_ELTWISE(0, 1, 240, 240, 256);
MACE_BM_ELTWISE(5, 1, 128, 128, 32);
MACE_BM_ELTWISE(5, 1, 240, 240, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
