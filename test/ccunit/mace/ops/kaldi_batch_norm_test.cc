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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class KaldiBatchNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const std::vector<index_t> &input_shape,
            const std::vector<float> &input_value,
            const int block_dim,
            const int dim,
            const std::vector<float> &scale,
            const std::vector<float> &offset,
            const std::vector<index_t> &output_shape,
            const std::vector<float> &output_value) {
  OpsTestNet net;
  int scale_dim = block_dim;
  if (scale_dim == -1) scale_dim = dim;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape,
                                  input_value);
  net.AddInputFromArray<D, float>("Scale", {scale_dim}, scale, true);
  net.AddInputFromArray<D, float>("Offset", {scale_dim}, offset, true);

  if (D == DeviceType::CPU) {
    OpDefBuilder("KaldiBatchNorm", "KaldiBatchNormOpTest")
        .Input("Input")
        .Input("Scale")
        .Input("Offset")
        .AddIntArg("block_dim", block_dim)
        .AddIntArg("test_mode", 1)
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run

    net.RunOp(D);
  } else if (D == DeviceType::GPU) {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = net.CreateTensor<float>(output_shape, output_value);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4);
}

template <DeviceType D>
void SimpleNotTestMode(const std::vector<index_t> &input_shape,
                       const std::vector<float> &input_value,
                       const int block_dim,
                       const std::vector<index_t> &output_shape,
                       const std::vector<float> &output_value) {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape,
                                  input_value);

  if (D == DeviceType::CPU) {
    OpDefBuilder("KaldiBatchNorm", "KaldiBatchNormOpTest")
        .Input("Input")
        .AddIntArg("block_dim", block_dim)
        .AddIntArg("test_mode", 0)
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run

    net.RunOp(D);
  } else if (D == DeviceType::GPU) {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = net.CreateTensor<float>(output_shape, output_value);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4);
}
}  // namespace

TEST_F(KaldiBatchNormOpTest, SimpleTestModeCPUOneBlock) {
  Simple<DeviceType::CPU>(
    {1, 6, 2},
    {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15},
    -1, 2,
    {4.f, 6.f},
    {2.f, 1.f},
    {1, 6, 2},
    {22, 31, 30, 43, 38, 55, 46, 67, 54, 79, 62, 91}); }

TEST_F(KaldiBatchNormOpTest, SimpleTestModeCPUTwoBlock) {
  Simple<DeviceType::CPU>(
    {1, 6, 4},
    {5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9,
    11, 11, 11, 11, 13, 13, 13, 13, 15, 15, 15, 15},
    2, 4,
    {4.f, 6.f},
    {2.f, 1.f},
    {1, 6, 4},
    {22, 31, 22, 31, 30, 43, 30, 43, 38, 55, 38, 55,
    46, 67, 46, 67, 54, 79, 54, 79, 62, 91, 62, 91});
}

TEST_F(KaldiBatchNormOpTest, SimpleNotTestModeCPUTwoBlock) {
  SimpleNotTestMode<DeviceType::CPU>(
    {1, 6, 4},
    {5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9,
    11, 11, 11, 11, 13, 13, 13, 13, 15, 15, 15, 15},
    2,
    {1, 6, 4},
    {-1.46379, -1.46379, -1.46379, -1.46379,
    -0.8783, -0.8783, -0.8783, -0.8783,
    -0.29276, -0.29276, -0.29276, -0.29276,
    0.29276, 0.29276, 0.29276, 0.29276,
    0.8783, 0.8783, 0.8783, 0.8783,
    1.46379, 1.46379, 1.46379, 1.46379});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
