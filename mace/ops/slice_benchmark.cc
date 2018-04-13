//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template<DeviceType D, typename T>
void BMSliceHelper(int iters,
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

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);

    auto builder = OpDefBuilder("Slice", "SliceTest");
    builder.Input("InputImage");
    for (int i = 0; i < num_outputs; ++i) {
      builder = builder.Output(MakeString("OutputImage", i));
    }
    builder
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    auto builder = OpDefBuilder("Slice", "SliceTest");
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

#define BM_SLICE_MACRO(N, H, W, C, NO, TYPE, DEVICE)                         \
  static void                                                                \
      BM_SLICE_##N##_##H##_##W##_##C##_##NO##_##TYPE##_##DEVICE(int iters) { \
        const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;     \
        mace::testing::MaccProcessed(tot);                                   \
        mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
        BMSliceHelper<DEVICE, TYPE>(iters, {N, H, W, C}, NO);                \
      }                                                                      \
      BENCHMARK(BM_SLICE_##N##_##H##_##W##_##C##_##NO##_##TYPE##_##DEVICE)

#define BM_SLICE(N, H, W, C, NO)                 \
  BM_SLICE_MACRO(N, H, W, C, NO, float, CPU);    \
  BM_SLICE_MACRO(N, H, W, C, NO, float, OPENCL); \
  BM_SLICE_MACRO(N, H, W, C, NO, half, OPENCL);

BM_SLICE(1, 32, 32, 32, 2);
BM_SLICE(1, 32, 32, 128, 2);
BM_SLICE(1, 32, 32, 256, 2);
BM_SLICE(1, 128, 128, 32, 2);
BM_SLICE(1, 128, 128, 128, 2);

}  // namespace test
}  // namespace ops
}  // namespace mace
