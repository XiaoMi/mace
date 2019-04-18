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

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void ConcatHelper(int iters, int concat_dim, int dim0, int dim1) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("Concat", "ConcatBM")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("axis", concat_dim)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, T>("Input0", {dim0, dim1});
  net.AddRandomInput<DeviceType::CPU, T>("Input1", {dim0, dim1});

  net.Setup(D);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Input0")->SetScale(0.1);
    net.GetTensor("Input1")->SetScale(0.2);
    net.GetTensor("Output")->SetScale(0.3);
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }
  const int64_t tot = static_cast<int64_t>(iters) * dim0 * dim1 * 2;
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
}
}  // namespace

#define MACE_BM_CONCAT_CPU_MACRO(AXIS, DIM0, DIM1, TYPE)                       \
  static void MACE_BM_CONCAT_CPU_##AXIS##_##DIM0##_##DIM1##_##TYPE(int iters) {\
    ConcatHelper<DeviceType::CPU, TYPE>(iters, AXIS, DIM0, DIM1);              \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_CONCAT_CPU_##AXIS##_##DIM0##_##DIM1##_##TYPE)

#ifdef MACE_ENABLE_QUANTIZE
#define MACE_BM_CONCAT_CPU(AXIS, DIM0, DIM1)             \
  MACE_BM_CONCAT_CPU_MACRO(AXIS, DIM0, DIM1, float);     \
  MACE_BM_CONCAT_CPU_MACRO(AXIS, DIM0, DIM1, uint8_t)
#else
#define MACE_BM_CONCAT_CPU(AXIS, DIM0, DIM1)             \
  MACE_BM_CONCAT_CPU_MACRO(AXIS, DIM0, DIM1, float)
#endif

MACE_BM_CONCAT_CPU(0, 100, 1000);
MACE_BM_CONCAT_CPU(0, 100, 100000);
MACE_BM_CONCAT_CPU(1, 100, 1000);
MACE_BM_CONCAT_CPU(1, 100, 100000);
MACE_BM_CONCAT_CPU(1, 1225, 128);

#ifdef MACE_ENABLE_OPENCL
namespace {
template <typename T>
void OpenCLConcatHelper(int iters,
                        const std::vector<index_t> &shape0,
                        const std::vector<index_t> &shape1,
                        int concat_dim) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input0", shape0);
  net.AddRandomInput<DeviceType::GPU, float>("Input1", shape1);

  OpDefBuilder("Concat", "ConcatBM")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("axis", concat_dim)
      .AddIntArg("has_data_format", 1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(DeviceType::GPU);
  }

  const int64_t tot =
      static_cast<int64_t>(iters) *
      (net.GetTensor("Input0")->size() + net.GetTensor("Input1")->size());
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(DeviceType::GPU);
  }
}
}  // namespace

#define MACE_BM_CONCAT_OPENCL_MACRO(N, H, W, C, TYPE)                          \
  static void MACE_BM_CONCAT_OPENCL_##N##_##H##_##W##_##C##_##TYPE(int iters) {\
    std::vector<index_t> shape = {N, H, W, C};                                 \
    OpenCLConcatHelper<TYPE>(iters, shape, shape, 3);                          \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_CONCAT_OPENCL_##N##_##H##_##W##_##C##_##TYPE)

MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 32, float);
MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 64, float);
MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 128, float);
MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 256, float);

MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 32, half);
MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 64, half);
MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 128, half);
MACE_BM_CONCAT_OPENCL_MACRO(3, 32, 32, 256, half);

#endif  // MACE_ENABLE_OPENCL

}  // namespace test
}  // namespace ops
}  // namespace mace
