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
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ProposalOpTest : public OpsTestBase {};

TEST_F(ProposalOpTest, CPUSimple) {
  const int img_height = 256;
  const int img_width = 256;
  const int height = 3;
  const int width = 4;

  OpsTestNet net;

  OpDefBuilder("Proposal", "ProposalTest")
      .Input("RpnCLSProb")
      .Input("RpnBBoxPred")
      .Input("ImgInfo")
      .AddIntArg("min_size", 16)
      .AddFloatArg("nms_thresh", 0.7)
      .AddIntArg("pre_nms_top_n", 12000)
      .AddIntArg("post_nms_top_n", 2000)
      .AddIntArg("feat_stride", 16)
      .AddIntArg("base_size", 16)
      .AddIntsArg("scales", {8, 16, 32})
      .AddFloatsArg("ratios", {0.5, 1, 2})
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  std::vector<float> scores(height * width * 18);
  for (size_t i = 0; i < scores.size(); ++i) {
    scores[i] = i;
  }

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>("RpnCLSProb",
                                                {1, height, width, 18}, scores);
  net.AddRepeatedInput<DeviceType::CPU, float>("RpnBBoxPred",
                                               {1, height, width, 4 * 9}, 1);
  net.AddInputFromArray<DeviceType::CPU, float>("ImgInfo", {1, 1, 1, 3},
                                                {img_height, img_width, 2});

  // Run
  net.RunOp();

  auto expected_tensor = CreateTensor<float>({1, 1, 1, 5}, {0, 0, 0, 255, 255});

  ExpectTensorNear<float>(*expected_tensor, *net.GetTensor("Output"), 1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
