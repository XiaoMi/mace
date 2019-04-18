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

#ifdef MACE_ENABLE_OPENCL

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void FilterBufferToImage(int iters,
                         int out_channel, int in_channel,
                         int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpContext context(net.ws(),
                    OpTestContext::Get()->GetDevice(DeviceType::GPU));

  // Add input data
  net.AddRandomInput<D, T>("Input",
                           {out_channel, in_channel, height, width});
  // Create output
  Tensor *b2i_output = net.ws()->CreateTensor(
      "B2IOutput", context.device()->allocator(), DataTypeToEnum<T>::value);

  auto transform_func = [&]() {
    OpenCLBufferTransformer<T>(MemoryType::GPU_BUFFER, MemoryType::GPU_IMAGE)
        .Transform(&context,
                   net.ws()->GetTensor("Input"),
                   OpenCLBufferType::IN_OUT_CHANNEL,
                   MemoryType::GPU_IMAGE,
                   0,
                   b2i_output);
  };

  for (int i = 0; i < 5; ++i) {
    transform_func();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    transform_func();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_B2I_MACRO(O, I, H, W, TYPE, DEVICE)                  \
  static void MACE_BM_B2I_##O##_##I##_##H##_##W##_##TYPE##_##DEVICE( \
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * O * I * H * W; \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    FilterBufferToImage<DEVICE, TYPE>(iters, O, I, H, W);            \
  }                                                                  \
  MACE_BENCHMARK(MACE_BM_B2I_##O##_##I##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_B2I(O, I, H, W)              \
  MACE_BM_B2I_MACRO(O, I, H, W, float, GPU); \
  MACE_BM_B2I_MACRO(O, I, H, W, half, GPU);

MACE_BM_B2I(5, 3, 3, 3);
MACE_BM_B2I(5, 3, 7, 7);
MACE_BM_B2I(32, 16, 1, 1);
MACE_BM_B2I(32, 16, 3, 3);
MACE_BM_B2I(32, 16, 5, 5);
MACE_BM_B2I(32, 16, 7, 7);
MACE_BM_B2I(64, 32, 1, 1);
MACE_BM_B2I(64, 32, 3, 3);
MACE_BM_B2I(64, 32, 5, 5);
MACE_BM_B2I(64, 32, 7, 7);
MACE_BM_B2I(128, 64, 1, 1);
MACE_BM_B2I(128, 64, 3, 3);
MACE_BM_B2I(128, 32, 1, 1);
MACE_BM_B2I(128, 32, 3, 3);
MACE_BM_B2I(256, 32, 1, 1);
MACE_BM_B2I(256, 32, 3, 3);

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_ENABLE_OPENCL
