// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void SoftmaxBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  OpDefBuilder("Softmax", "SoftmaxBM")
      .Input("Input")
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

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

#ifdef MACE_ENABLE_QUANTIZE
template <>
void SoftmaxBenchmark<CPU, uint8_t>(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::CPU, uint8_t>(
      "Input", {batch, height, width, channels});

  OpDefBuilder("Softmax", "SoftmaxBM")
      .Input("Input")
      .Output("Output")
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  net.Setup(DeviceType::CPU);

  Tensor *output = net.GetTensor("Output");
  output->SetScale(0);
  output->SetZeroPoint(1);

  Tensor *input = net.GetTensor("Input");
  input->SetScale(0.1);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
#endif  // MACE_ENABLE_QUANTIZE

}  // namespace

#define MACE_BM_SOFTMAX_MACRO(N, C, H, W, TYPE, DEVICE)                   \
  static void MACE_BM_SOFTMAX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(  \
      int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    SoftmaxBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                    \
  }                                                                       \
  MACE_BENCHMARK(MACE_BM_SOFTMAX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#if defined(MACE_ENABLE_OPENCL) && defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_SOFTMAX(N, C, H, W)                 \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, float, CPU);    \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, uint8_t, CPU);  \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, float, GPU);    \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, half, GPU)
#elif defined(MACE_ENABLE_OPENCL)
#define MACE_BM_SOFTMAX(N, C, H, W)                 \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, float, CPU);    \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, float, GPU);    \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, half, GPU)
#elif defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_SOFTMAX(N, C, H, W)                 \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, float, CPU);    \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, uint8_t, CPU)
#else
#define MACE_BM_SOFTMAX(N, C, H, W)                 \
  MACE_BM_SOFTMAX_MACRO(N, C, H, W, float, CPU)
#endif

MACE_BM_SOFTMAX(1, 2, 512, 512);
MACE_BM_SOFTMAX(1, 3, 512, 512);
MACE_BM_SOFTMAX(1, 4, 512, 512);
MACE_BM_SOFTMAX(1, 10, 256, 256);
MACE_BM_SOFTMAX(1, 1024, 7, 7);

}  // namespace test
}  // namespace ops
}  // namespace mace
