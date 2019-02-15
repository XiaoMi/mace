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

#include "gmock/gmock.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class PriorBoxOpTest : public OpsTestBase {};

TEST_F(PriorBoxOpTest, Simple) {
  OpsTestNet net;
  //  Add input data
  net.AddRandomInput<DeviceType::CPU, float>("INPUT", {1, 128, 1, 1});
  net.AddRandomInput<DeviceType::CPU, float>("DATA", {1, 3, 300, 300});
  OpDefBuilder("PriorBox", "PriorBoxTest")
      .Input("INPUT")
      .Input("DATA")
      .Output("OUTPUT")
      .AddFloatsArg("min_size", {285})
      .AddFloatsArg("max_size", {300})
      .AddFloatsArg("aspect_ratio", {1, 2, 0.5, 3, 0.33333333333})
      .AddIntArg("clip", 0)
      .AddFloatsArg("variance", {0.1, 0.1, 0.2, 0.2})
      .AddFloatArg("offset", 0.5)
      .Finalize(net.NewOperatorDef());

  //  Run
  net.RunOp(DeviceType::CPU);

  //  Check
  auto expected_tensor = net.CreateTensor<float>({1, 2, 24},
      {0.025, 0.025, 0.975, 0.975,
       0.012660282759551838, 0.012660282759551838,
       0.9873397172404482, 0.9873397172404482,
       -0.17175144212722018, 0.16412427893638995,
       1.1717514421272204, 0.8358757210636101,
       0.16412427893638995, -0.17175144212722018,
       0.8358757210636101, 1.1717514421272204,
       -0.3227241335952166, 0.22575862213492773,
       1.3227241335952165, 0.7742413778650723,
       0.22575862213492773, -0.3227241335952166,
       0.7742413778650723, 1.3227241335952165,
       0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2,
       0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2});
  ExpectTensorNear<float>(*expected_tensor, *net.GetTensor("OUTPUT"));
}
}  // namespace test
}  // namespace ops
}  // namespace mace
