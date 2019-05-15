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

class ArgMaxOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void ArgMaxTest(const std::vector<index_t> &input_shape,
                const std::vector<float> &input,
                const std::vector<index_t> &output_shape,
                const std::vector<int32_t> &output) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input);
  net.AddInputFromArray<D, int32_t>("axis", {}, {-1});

  if (D == DeviceType::CPU) {
    OpDefBuilder("ArgMax", "ArgMaxTest")
        .Input("Input")
        .Input("axis")
        .Output("Output")
        .OutputType({DT_INT32})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = net.CreateTensor<int32_t>(output_shape, output);
  ExpectTensorNear<int32_t>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ArgMaxOpTest, Vector) { ArgMaxTest<CPU>({3}, {-3, -1, -2}, {}, {1}); }

TEST_F(ArgMaxOpTest, Matrix) {
  ArgMaxTest<CPU>({3, 3}, {4, 5, 6, 9, 8, 7, 1, 2, 3}, {3}, {2, 0, 2});
}

TEST_F(ArgMaxOpTest, HighRank) {
  ArgMaxTest<CPU>({1, 2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                  {1, 2, 2}, {2, 2, 2, 2});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
