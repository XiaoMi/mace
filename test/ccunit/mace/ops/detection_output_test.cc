// Copyright 2021 The MACE Authors. All Rights Reserved.
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
#include "./detection_output_test_data.h"

namespace mace {
namespace ops {
namespace test {

class DetectionOutputTest : public OpsTestBase {};

namespace {
template <RuntimeType D>
void RunDetectionOutput(const std::vector<index_t> &input_shape1,
                        const std::vector<float> &input_data1,
                        const std::vector<index_t> &input_shape2,
                        const std::vector<float> &input_data2,
                        const std::vector<index_t> &input_shape3,
                        const std::vector<float> &input_data3,
                        const std::vcetor<int> &offset,
                        const int num_classes,
                        const float nms_threshold,
                        const int nms_top_k,
                        const int keep_top_k,
                        const float confidence_threshold,
                        const std::vector<index_t> &expected_shape,
                        const std::vector<float> &expected_data) {
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input1", input_shape1, input_data1);
  net.AddInputFromArray<D, float>("Input2", input_shape2, input_data2);
  net.AddInputFromArray<D, float>("Input3", input_shape3, input_data3);

  OpDefBuilder("DetectionOutput", "DetectionOutputTest")
      .Input("Input1")
      .Input("Input2")
      .Input("Input3")
      .Output("Output")
      .AddIntArg("num_classes", num_classes)
      .AddFloatArg("nms_threshold", nms_threshold)
      .AddIntArg("nms_top_k", nms_top_k)
      .AddIntArg("keep_top_k", keep_top_k)
      .AddFloatArg("confidence_threshold", confidence_threshold)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(D);

  // Check
  auto expected = net.CreateTensor<float>(expected_shape, expected_data);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"));
}
}  // namespace

TEST_F(DetectionOutputTest, SimpleCPU) {
  auto *f_mbox_loc = reinterpret_cast<float>(mbox_loc);
  auto *f_mbox_conf_flatten = reinterpret_cast<float>(mbox_conf_flatten);
  auto *f_mbox_priorbox = reinterpret_cast<float>(mbox_priorbox);

  RunDetectionOutput<RuntimeType::RT_CPU>(
      {1, 57120},
      std::vector<float>(f_mbox_loc, f_mbox_loc + mbox_loc_len / sizeof(float)),
      {1, 28560},
      std::vector<float>(
          f_mbox_conf_flatten,
          f_mbox_conf_flatten + mbox_conf_flatten_len / sizeof(float)),
      {1, 2, 57120},
      std::vector<float>(f_mbox_priorbox,
                         f_mbox_priorbox + mbox_priorbox_len / sizeof(float)),
      2, 0.3f, 400, 3, 0.05,
      {1, 1, 3, 7},
      {0., 1., 0.9999993,  0.13859153,  0.48088723,  0.71894324, 0.8581148,
       0., 1., 0.10084193, 0.30128145,  -0.06979306, 0.7539885,  0.17274505,
       0., 1., 0.02140488, -0.01885228, -0.01132634, 0.04337814, 0.02579039});
}


}  // namespace test
}  // namespace ops
}  // namespace mace
