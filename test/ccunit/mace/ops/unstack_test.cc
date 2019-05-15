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

class UnstackOpTest : public OpsTestBase {};

namespace {

void TestUnstack(const std::vector<index_t> &input_shape,
                 const std::vector<float> &input,
                 int axis,
                 const std::vector<index_t> &output_shape,
                 const std::vector<std::vector<float>> &outputs) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input", input_shape, input);

  auto op_builder = OpDefBuilder("Unstack", "UnstackOpTest")
                        .Input("Input")
                        .AddIntArg("axis", axis);

  for (size_t i = 0; i < outputs.size(); ++i) {
    op_builder.Output(MakeString("Output", i));
  }
  op_builder.Finalize(net.NewOperatorDef());

  net.RunOp();

  for (size_t i = 0; i < outputs.size(); ++i) {
    net.AddInputFromArray<CPU, float>("ExpectedOutput", output_shape,
                                      outputs[i]);
    ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                            *net.GetOutput(MakeString("Output", i).c_str()));
  }
}

}  // namespace

TEST_F(UnstackOpTest, TestUnstackScalar) {
  TestUnstack({3}, {1, 2, 3}, 0, {}, {{1}, {2}, {3}});
}

TEST_F(UnstackOpTest, TestUnstackVector) {
  TestUnstack({3, 2}, {1, 4, 2, 5, 3, 6}, 0, {2}, {{1, 4}, {2, 5}, {3, 6}});
  TestUnstack({3, 2}, {1, 4, 2, 5, 3, 6}, -2, {2}, {{1, 4}, {2, 5}, {3, 6}});
  TestUnstack({2, 3}, {1, 2, 3, 4, 5, 6}, 1, {2}, {{1, 4}, {2, 5}, {3, 6}});
}

TEST_F(UnstackOpTest, TestUnstackHighRank) {
  TestUnstack({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, -3, {2, 3},
      {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}});
  TestUnstack({2, 2, 3}, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}, 1, {2, 3},
      {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}});
  TestUnstack({2, 3, 2}, {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}, 2, {2, 3},
      {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
