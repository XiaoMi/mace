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

class CropTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void RunCrop(const std::vector<index_t> &input_shape,
             const std::vector<float> &input_data,
             const std::vector<index_t> &input_shape2,
             const std::vector<int> &offset,
             const std::vector<index_t> &expected_shape,
             const std::vector<float> &expected_data) {
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input0", input_shape, input_data);
  net.AddRandomInput<D, float>("Input1", input_shape2);

  if (D == GPU) {
    OpDefBuilder("Crop", "CropTest")
        .Input("Input0")
        .Input("Input1")
        .Output("Output")
        .AddIntsArg("offset", offset)
        .AddIntArg("has_data_format", 1)
        .Finalize(net.NewOperatorDef());
  } else if (D == CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input0",
                                                    DataFormat::NHWC,
                                                    "InputNCHW0",
                                                    DataFormat::NCHW);
    net.TransformDataFormat<DeviceType::CPU, float>("Input1",
                                                    DataFormat::NHWC,
                                                    "InputNCHW1",
                                                    DataFormat::NCHW);
    OpDefBuilder("Crop", "CropTest")
        .Input("InputNCHW0")
        .Input("InputNCHW1")
        .Output("OutputNCHW")
        .AddIntsArg("offset", offset)
        .AddIntArg("has_data_format", 1)
        .Finalize(net.NewOperatorDef());
  }

  // Run
  net.RunOp(D);

  if (D == CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }
  // Check
  auto expected = net.CreateTensor<float>(expected_shape, expected_data);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"));
}
}  // namespace

TEST_F(CropTest, SimpleCPU) {
  RunCrop<DeviceType::CPU>({1, 10, 10, 3},
                           {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0}, {1, 5, 5, 3}, {-1, 2, 2, -1},
                            {1, 5, 5, 3},
                            {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0});
}

TEST_F(CropTest, SimpleGPU) {
  RunCrop<DeviceType::GPU>({1, 10, 10, 3},
                            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0}, {1, 5, 5, 3}, {-1, 2, 2, -1},
                            {1, 5, 5, 3},
                            {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 3.0, 3.0, 3.0});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
