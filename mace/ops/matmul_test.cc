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

#include <fstream>

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class MatMulOpTest : public OpsTestBase {};

namespace {
template<DeviceType D>
void Simple(const std::vector<index_t> &A_shape,
            const std::vector<float> &A_value,
            const std::vector<index_t> &B_shape,
            const std::vector<float> &B_value,
            const std::vector<index_t> &C_shape,
            const std::vector<float> &C_value) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("A", A_shape, A_value);
  net.AddInputFromArray<D, float>("B", B_shape, B_value);

  OpDefBuilder("MatMul", "MatMulTest")
      .Input("A")
      .Input("B")
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);

  // Check
  auto expected = net.CreateTensor<float>(C_shape, C_value);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(MatMulOpTest, SimpleCPU) {
  Simple<DeviceType::CPU>({1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 3, 2},
                          {1, 2, 3, 4, 5, 6}, {1, 2, 2}, {22, 28, 49, 64});
  Simple<DeviceType::CPU>(
      {1, 5, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
      {1, 5, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
      {1, 5, 5}, {215, 230, 245, 260, 275, 490, 530, 570, 610,
                  650, 765, 830, 895, 960, 1025, 1040, 1130, 1220,
                  1310, 1400, 1315, 1430, 1545, 1660, 1775});
}

TEST_F(MatMulOpTest, SimpleCPUWithBatch) {
  Simple<DeviceType::CPU>({2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                          {2, 3, 2}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                          {2, 2, 2}, {22, 28, 49, 64, 22, 28, 49, 64});
}

namespace {
void QuantOutputUint8(const std::vector<index_t> &batch,
                      const index_t height,
                      const index_t channels,
                      const index_t out_width,
                      const bool transpose_a,
                      const bool transpose_b) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  index_t batch_count = std::accumulate(batch.begin(), batch.end(), 1,
                                        std::multiplies<index_t>());
  if (transpose_a) {
    net.AddRandomInput<CPU, float>("A", {batch_count, channels, height});
  } else {
    net.AddRandomInput<CPU, float>("A", {batch_count, height, channels});
  }
  if (transpose_b) {
    net.AddRandomInput<CPU, float>("B", {batch_count, out_width, channels});
  } else {
    net.AddRandomInput<CPU, float>("B", {batch_count, channels, out_width});
  }

  OpDefBuilder("MatMul", "MatMulTest")
      .Input("A")
      .AddIntArg("transpose_a", transpose_a ? 1 : 0)
      .Input("B")
      .AddIntArg("transpose_b", transpose_b ? 1 : 0)
      .Output("Output")
      .AddIntArg("T", DT_FLOAT)
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);

  OpDefBuilder("Quantize", "QuantizeA")
      .Input("A")
      .Output("QuantizedA")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeB")
      .Input("B")
      .Output("QuantizedB")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeOutput")
      .Input("Output")
      .Output("ExpectedQuantizedOutput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("MatMul", "QuantizeMatMulTest")
      .Input("QuantizedA")
      .AddIntArg("transpose_a", transpose_a ? 1 : 0)
      .Input("QuantizedB")
      .AddIntArg("transpose_b", transpose_b ? 1 : 0)
      .Output("QuantizedOutput")
      .AddIntArg("T", DT_UINT8)
      .OutputType({DT_UINT8})
      .Finalize(net.NewOperatorDef());
  net.Setup(DeviceType::CPU);
  Tensor *eq_output = net.GetTensor("ExpectedQuantizedOutput");
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(eq_output->scale());
  q_output->SetZeroPoint(eq_output->zero_point());
  net.Run();

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}

void QuantOutputInt32(const std::vector<index_t> &batch,
                      const index_t height,
                      const index_t channels,
                      const index_t out_width,
                      const bool transpose_a,
                      const bool transpose_b) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  index_t batch_count = std::accumulate(batch.begin(), batch.end(), 1,
                                        std::multiplies<index_t>());
  if (transpose_a) {
    net.AddRandomInput<CPU, float>("A", {batch_count, channels, height},
                                   false);
  } else {
    net.AddRandomInput<CPU, float>("A", {batch_count, height, channels},
                                   false);
  }
  if (transpose_b) {
    net.AddRandomInput<CPU, float>("B",
                                   {batch_count, out_width, channels},
                                   false);
  } else {
    net.AddRandomInput<CPU, float>("B",
                                   {batch_count, channels, out_width},
                                   false);
  }

  OpDefBuilder("MatMul", "MatMulTest")
      .Input("A")
      .AddIntArg("transpose_a", transpose_a ? 1 : 0)
      .Input("B")
      .AddIntArg("transpose_b", transpose_b ? 1 : 0)
      .Output("Output")
      .AddIntArg("T", DT_FLOAT)
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);

  OpDefBuilder("Quantize", "QuantizeA")
      .Input("A")
      .Output("QuantizedA")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeB")
      .Input("B")
      .Output("QuantizedB")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("MatMul", "QuantizeMatMulTest")
      .Input("QuantizedA")
      .AddIntArg("transpose_a", transpose_a ? 1 : 0)
      .Input("QuantizedB")
      .AddIntArg("transpose_b", transpose_b ? 1 : 0)
      .Output("QuantizedOutput")
      .AddIntArg("T", DT_UINT8)
      .OutputType({DT_INT32})
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_INT32)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}
}  // namespace

TEST_F(MatMulOpTest, QuantOutputUint8) {
  QuantOutputUint8({1}, 64, 128, 32, false, false);
  QuantOutputUint8({1}, 64, 32, 128, false, false);
  QuantOutputUint8({2, 3}, 64, 32, 128, false, false);
  QuantOutputUint8({1}, 64, 128, 32, false, true);
  QuantOutputUint8({1}, 64, 32, 128, false, true);
  QuantOutputUint8({2, 3}, 64, 32, 128, false, true);
  QuantOutputUint8({1}, 64, 128, 32, true, false);
  QuantOutputUint8({1}, 64, 32, 128, true, false);
  QuantOutputUint8({2, 3}, 64, 32, 128, true, false);
  QuantOutputUint8({1}, 64, 128, 32, true, true);
  QuantOutputUint8({1}, 64, 32, 128, true, true);
  QuantOutputUint8({2, 3}, 64, 32, 128, true, true);
  // UnAligned
  QuantOutputUint8({2}, 3, 3, 3, false, false);
  QuantOutputUint8({16}, 31, 61, 67, false, true);
  QuantOutputUint8({31}, 31, 61, 67, true, false);
  QuantOutputUint8({2, 3}, 31, 61, 67, true, true);
}

TEST_F(MatMulOpTest, QuantOutputInt32) {
  QuantOutputInt32({1}, 64, 128, 32, false, false);
  QuantOutputInt32({1}, 64, 32, 128, false, false);
  QuantOutputInt32({2, 3}, 64, 32, 128, false, false);
  QuantOutputInt32({1}, 64, 128, 32, false, true);
  QuantOutputInt32({1}, 64, 32, 128, false, true);
  QuantOutputInt32({2, 3}, 64, 32, 128, false, true);
  QuantOutputInt32({1}, 64, 128, 32, true, false);
  QuantOutputInt32({1}, 64, 32, 128, true, false);
  QuantOutputInt32({2, 3}, 64, 32, 128, true, false);
  QuantOutputInt32({1}, 64, 128, 32, true, true);
  QuantOutputInt32({1}, 64, 32, 128, true, true);
  QuantOutputInt32({2, 3}, 64, 32, 128, true, true);
  QuantOutputInt32({1}, 1, 30000, 256, false, true);
  QuantOutputInt32({1}, 30000, 256, 1, false, false);
  QuantOutputInt32({2}, 1, 256, 128, false, true);
  QuantOutputInt32({3}, 128, 256, 1, false, false);

  // UnAligned
  QuantOutputInt32({2}, 3, 3, 3, false, false);
  QuantOutputInt32({16}, 31, 61, 67, false, true);
  QuantOutputInt32({31}, 31, 61, 67, true, false);
  QuantOutputInt32({2, 3}, 31, 61, 67, true, true);
  QuantOutputInt32({1}, 1, 30001, 253, false, true);
  QuantOutputInt32({2}, 253, 300, 1, false, false);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
