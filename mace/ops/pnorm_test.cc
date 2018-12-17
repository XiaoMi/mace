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

class PNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestPNorm(const std::vector<index_t> &input_shape,
               const std::vector<T> &input,
               const int p,
               const int output_dim,
               const std::vector<index_t> &output_shape,
               const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);

  OpDefBuilder("PNorm", "PNormTest")
      .Input("Input")
      .AddIntArg("p", p)
      .AddIntArg("output_dim", output_dim)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, T>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(PNormOpTest, SimpleTest) {
  TestPNorm<DeviceType::CPU, float>(
    {1, 5, 10},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
    2, 5,
    {1, 5, 5},
    {2.236067977, 5, 7.810249676, 10.630145813, 13.453624047,
     5, 7.810249676, 10.630145813, 13.453624047, 16.278820596,
     7.810249676, 10.630145813, 13.453624047, 16.278820596, 19.104973175,
     10.630145813, 13.453624047, 16.278820596, 19.104973175, 21.931712199,
     13.453624047, 16.278820596, 19.104973175, 21.931712199, 24.758836806});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
