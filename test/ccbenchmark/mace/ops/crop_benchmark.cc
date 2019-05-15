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
void CropHelper(int iters,
                const std::vector<index_t> &shape0,
                const std::vector<index_t> &shape1,
                int crop_axis,
                int offset) {
  mace::testing::StopTiming();

  OpsTestNet net;

  std::vector<int> offsets(4, -1);

  for (int i = crop_axis; i < 4; ++i) {
    offsets[i] = offset;
  }

  if (D == DeviceType::CPU) {
    auto input_shape0 = TransposeShape<index_t, index_t>(shape0, {0, 3, 1, 2});
    auto input_shape1 = TransposeShape<index_t, index_t>(shape1, {0, 3, 1, 2});
    net.AddRandomInput<D, float>("Input0", input_shape0);
    net.AddRandomInput<D, float>("Input1", input_shape1);
#ifdef MACE_ENABLE_OPENCL
  } else if (D == DeviceType::GPU) {
    // Add input data
    net.AddRandomInput<D, T>("Input0", shape0);
    net.AddRandomInput<D, T>("Input1", shape1);
#endif  // MACE_ENABLE_OPENCL
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  OpDefBuilder("Crop", "CropBM")
      .Input("Input0")
      .Input("Input1")
      .AddIntsArg("offset", offsets)
      .AddIntArg("has_data_format", 1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  net.Setup(D);
  for (int i = 0; i < 1; ++i) {
    net.Run();
  }

  const int64_t tot =
      static_cast<int64_t>(iters) *
      (net.GetTensor("Input0")->size());
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
}
}  // namespace

#define MACE_BM_CROP_MACRO(N, H, W, C, AXIS, OFFSET, DEVICE, TYPE)     \
  static void MACE_BM_CROP_##N##_##H##_##W##_##C##_##AXIS##_##OFFSET## \
  _##DEVICE##_##TYPE(int iters) {                                      \
    std::vector<index_t> shape0 = {N, H, W, C};                        \
    std::vector<index_t> shape1 = {N / 2, H / 2, W / 2, C / 2};        \
    CropHelper<DEVICE, TYPE>(iters, shape0, shape1, AXIS, OFFSET);     \
  }                                                                    \
  MACE_BENCHMARK(MACE_BM_CROP_##N##_##H##_##W##_##C##_##AXIS##_##OFFSET\
  ##_##DEVICE##_##TYPE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_CROP(N, H, W, C, AXIS, OFFSET)               \
  MACE_BM_CROP_MACRO(N, H, W, C, AXIS, OFFSET, CPU, float);  \
  MACE_BM_CROP_MACRO(N, H, W, C, AXIS, OFFSET, GPU, float);  \
  MACE_BM_CROP_MACRO(N, H, W, C, AXIS, OFFSET, GPU, half)
#else
#define MACE_BM_CROP(N, H, W, C, AXIS, OFFSET)               \
  MACE_BM_CROP_MACRO(N, H, W, C, AXIS, OFFSET, CPU, float)
#endif  // MACE_ENABLE_OPENCL

MACE_BM_CROP(4, 32, 32, 32, 2, 4);
MACE_BM_CROP(8, 32, 32, 64, 1, 0);
MACE_BM_CROP(8, 32, 32, 128, 0, 0);
MACE_BM_CROP(8, 32, 32, 256, 2, 4);

}  // namespace test
}  // namespace ops
}  // namespace mace
