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

class TransposeOpTest : public OpsTestBase {};

namespace {
void TransposeNCHWTest(const std::vector<index_t> &input_shape) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<CPU, float>("Input", input_shape);

  OpDefBuilder("Transpose", "TransposeNCHWTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("dims", {0, 3, 1, 2})
      .Finalize(net.NewOperatorDef());

  // Run on cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  ExpectTensorNear<float>(*net.GetOutput("InputNCHW"),
                          *net.GetOutput("Output"));
}

void TransposeNHWCTest(const std::vector<index_t> &input_shape) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<CPU, float>("Input", input_shape);

  OpDefBuilder("Transpose", "TransposeNHWCTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("dims", {0, 2, 3, 1})
      .Finalize(net.NewOperatorDef());

  // Run on cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NCHW, "InputNHWC", DataFormat::NHWC);

  ExpectTensorNear<float>(*net.GetOutput("InputNHWC"),
                          *net.GetOutput("Output"));
}
}  // namespace

TEST_F(TransposeOpTest, NHWC_to_NCHW) {
  TransposeNCHWTest({3, 64, 64, 128});
  TransposeNCHWTest({1, 64, 48, 128});
  TransposeNCHWTest({1, 512, 512, 3});
  TransposeNCHWTest({2, 512, 512, 3});
}

TEST_F(TransposeOpTest, NCHW_to_NHWC) {
  TransposeNHWCTest({1, 2, 512, 512});
  TransposeNHWCTest({1, 3, 512, 512});
  TransposeNHWCTest({2, 2, 512, 512});
}

TEST_F(TransposeOpTest, Rank2) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<CPU, float>("Input", {2, 3}, {1, 2, 3, 4, 5, 6});

  OpDefBuilder("Transpose", "TransposeNCHWTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("dims", {1, 0})
      .Finalize(net.NewOperatorDef());

  // Run on cpu
  net.RunOp();

  net.AddInputFromArray<CPU, float>("ExpectedOutput", {3, 2},
                                    {1, 4, 2, 5, 3, 6});

  ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                          *net.GetOutput("Output"));
}

}  // namespace test
}  // namespace ops
}  // namespace mace
