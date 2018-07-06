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
template <DeviceType D, typename T>
void CropHelper(int iters, int crop_axis, int dim1, int offset) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("Crop", "CropBM")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("axis", crop_axis)
      .AddIntsArg("offset", {offset})
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  const int kDim0 = 100;
  net.AddRandomInput<DeviceType::CPU, T>("Input0", {1, kDim0, dim1, dim1, });
  net.AddRandomInput<DeviceType::CPU, T>("Input1",
                                         {1, kDim0 / 2, dim1 / 2, dim1 / 2});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  const int64_t tot = static_cast<int64_t>(iters) * kDim0 * dim1 * dim1;
  mace::testing::MaccProcessed(tot);
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}
}  // namespace

#define MACE_BM_CROP_CPU_MACRO(AXIS, DIM, OFFSET)                     \
  static void MACE_BM_CROP_CPU_##AXIS##_##DIM##_##OFFSET(int iters) { \
    CropHelper<DeviceType::CPU, float>(iters, AXIS, DIM, OFFSET);     \
  }                                                               \
  MACE_BENCHMARK(MACE_BM_CROP_CPU_##AXIS##_##DIM##_##OFFSET)

MACE_BM_CROP_CPU_MACRO(1, 256, 3);
MACE_BM_CROP_CPU_MACRO(2, 256, 3);
MACE_BM_CROP_CPU_MACRO(3, 512, 3);
MACE_BM_CROP_CPU_MACRO(2, 512, 6);

namespace {
template <typename T>
void OpenclCropHelper(int iters,
                      const std::vector<index_t> &shape0,
                      const std::vector<index_t> &shape1,
                      int crop_axis,
                      int offset) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input0", shape0);
  net.AddRandomInput<DeviceType::GPU, float>("Input1", shape1);

  BufferToImage<DeviceType::GPU, T>(&net, "Input0", "InputImage0",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::GPU, T>(&net, "Input1", "InputImage1",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("Crop", "CropBM")
      .Input("InputImage0")
      .Input("InputImage1")
      .AddIntArg("axis", crop_axis)
      .AddIntsArg("offset", {offset})
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(DeviceType::GPU);
  }

  const int64_t tot =
      static_cast<int64_t>(iters) *
      (net.GetTensor("Input0")->size() + net.GetTensor("Input1")->size());
  mace::testing::MaccProcessed(tot);
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(DeviceType::GPU);
  }
}
}  // namespace

#define MACE_BM_CROP_GPU_MACRO(N, H, W, C, AXIS, OFFSET, TYPE)            \
  static void MACE_BM_CROP_GPU_##N##_##H##_##W##_##C##_##AXIS##_##OFFSET##\
  _##TYPE(int iters) {                                                        \
    std::vector<index_t> shape0 = {N, H, W, C};                              \
    std::vector<index_t> shape1 = {N / 2, H / 2, W / 2, C / 2};              \
    OpenclCropHelper<TYPE>(iters, shape0, shape1, AXIS, OFFSET);             \
  }                                                                          \
  MACE_BENCHMARK(MACE_BM_CROP_GPU_##N##_##H##_##W##_##C##_##AXIS##_##OFFSET\
  ##_##TYPE)

MACE_BM_CROP_GPU_MACRO(4, 32, 32, 32, 2, 4, float);
MACE_BM_CROP_GPU_MACRO(8, 32, 32, 64, 1, 0, float);
MACE_BM_CROP_GPU_MACRO(8, 32, 32, 128, 0, 0, float);
MACE_BM_CROP_GPU_MACRO(8, 32, 32, 256, 2, 4, float);

MACE_BM_CROP_GPU_MACRO(4, 32, 32, 32, 2, 4, half);
MACE_BM_CROP_GPU_MACRO(8, 32, 32, 64, 1, 0, half);
MACE_BM_CROP_GPU_MACRO(8, 32, 32, 128, 0, 0, half);
MACE_BM_CROP_GPU_MACRO(8, 32, 32, 256, 2, 4, half);

}  // namespace test
}  // namespace ops
}  // namespace mace
