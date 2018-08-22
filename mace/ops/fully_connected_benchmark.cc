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
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void FCBenchmark(
    int iters, int batch, int height, int width, int channel, int out_channel) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channel});
  net.AddRandomInput<D, float>("Weight",
                               {out_channel, channel, height, width});
  net.AddRandomInput<D, float>("Bias", {out_channel});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input",
                                                    NHWC,
                                                    "InputNCHW",
                                                    NCHW);
    OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("InputNCHW")
      .Input("Weight")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  } else if (D == DeviceType::GPU) {
    kernels::BufferType weight_type = kernels::BufferType::WEIGHT_WIDTH;
    BufferToImage<D, T>(&net, "Weight", "WeightImage",
                        weight_type);
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);

    OpDefBuilder("FullyConnected", "FullyConnectedTest")
        .Input("InputImage")
        .Input("WeightImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntArg("weight_type", static_cast<int>(weight_type))
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
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

template <>
void FCBenchmark<CPU, uint8_t>(
    int iters, int batch, int height, int width, int channel, int out_channel) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<CPU, uint8_t>("Input", {batch, height, width, channel});
  net.GetTensor("Input")->SetScale(0.1);
  net.AddRandomInput<CPU, uint8_t>("Weight",
                                   {out_channel, height, width, channel});
  net.GetTensor("Weight")->SetScale(0.1);
  net.AddRandomInput<CPU, uint8_t>("Bias", {out_channel});

  OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("Input")
      .Input("Weight")
      .Input("Bias")
      .Output("Output")
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  net.Setup(CPU);
  net.GetTensor("Output")->SetScale(0.1);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
}
}  // namespace

#define MACE_BM_FC_MACRO(N, H, W, C, OC, TYPE, DEVICE)                     \
  static void MACE_BM_FC_##N##_##H##_##W##_##C##_##OC##_##TYPE##_##DEVICE( \
      int iters) {                                                         \
    const int64_t macc =                                                   \
        static_cast<int64_t>(iters) * N * C * H * W * OC + OC;             \
    const int64_t tot =                                                    \
        static_cast<int64_t>(iters) * (N + OC) * C * H * W + OC;           \
    mace::testing::MaccProcessed(macc);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                    \
    FCBenchmark<DEVICE, TYPE>(iters, N, H, W, C, OC);                      \
  }                                                                        \
  MACE_BENCHMARK(MACE_BM_FC_##N##_##H##_##W##_##C##_##OC##_##TYPE##_##DEVICE)

#define MACE_BM_FC(N, H, W, C, OC)                 \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, CPU);    \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, GPU);    \
  MACE_BM_FC_MACRO(N, H, W, C, OC, half, GPU);     \
  MACE_BM_FC_MACRO(N, H, W, C, OC, uint8_t, CPU);

MACE_BM_FC(1, 16, 16, 32, 32);
MACE_BM_FC(1, 8, 8, 32, 1000);
MACE_BM_FC(1, 2, 2, 512, 2);
MACE_BM_FC(1, 7, 7, 512, 2048);

}  // namespace test
}  // namespace ops
}  // namespace mace
