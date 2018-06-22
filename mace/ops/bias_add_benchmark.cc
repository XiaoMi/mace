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
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void BiasAdd(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, T>("Input", {batch, channels, height, width});
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, T>("Input", {batch, height, width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  net.AddRandomInput<D, T>("Bias", {channels}, true);

  if (D == DeviceType::CPU) {
    OpDefBuilder("BiasAdd", "BiasAddBM")
      .Input("Input")
      .Input("Bias")
      .AddIntArg("data_format", NCHW)
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("BiasAdd", "BiasAddBM")
        .Input("InputImage")
        .Input("BiasImage")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_BIAS_ADD_MACRO(N, C, H, W, TYPE, DEVICE)                  \
  static void MACE_BM_BIAS_ADD_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE( \
      int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::MaccProcessed(tot);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    BiasAdd<DEVICE, TYPE>(iters, N, C, H, W);                             \
  }                                                                       \
  MACE_BENCHMARK(MACE_BM_BIAS_ADD_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_BIAS_ADD(N, C, H, W)                 \
  MACE_BM_BIAS_ADD_MACRO(N, C, H, W, float, CPU);    \
  MACE_BM_BIAS_ADD_MACRO(N, C, H, W, float, GPU);    \
  MACE_BM_BIAS_ADD_MACRO(N, C, H, W, half, GPU);

MACE_BM_BIAS_ADD(1, 1, 512, 512);
MACE_BM_BIAS_ADD(1, 3, 128, 128);
MACE_BM_BIAS_ADD(1, 3, 512, 512);
MACE_BM_BIAS_ADD(1, 32, 112, 112);
MACE_BM_BIAS_ADD(1, 64, 256, 256);
MACE_BM_BIAS_ADD(1, 64, 512, 512);
MACE_BM_BIAS_ADD(1, 128, 56, 56);
MACE_BM_BIAS_ADD(1, 128, 256, 256);
MACE_BM_BIAS_ADD(1, 256, 14, 14);
MACE_BM_BIAS_ADD(1, 512, 14, 14);
MACE_BM_BIAS_ADD(1, 1024, 7, 7);
MACE_BM_BIAS_ADD(32, 1, 256, 256);
MACE_BM_BIAS_ADD(32, 3, 256, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
