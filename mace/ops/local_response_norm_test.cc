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

class LocalResponseNormOpTest : public OpsTestBase {};

template <DeviceType D>
void Simple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 1, 2, 6},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

    OpDefBuilder("LocalResponseNorm", "LocalResponseNormTest")
        .Input("InputNCHW")
        .AddIntArg("depth_radius", 5)
        .AddFloatArg("bias", 1.0f)
        .AddFloatArg("alpha", 1.0f)
        .AddFloatArg("beta", 0.5f)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 1, 2, 6},
      {0.28, 0.28, 0.39, 0.39, 0.51, 0.51, 0.34, 0.34, 0.40, 0.40, 0.47, 0.47});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0, 1e-2);
}

TEST_F(LocalResponseNormOpTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

}  // namespace test
}  // namespace ops
}  // namespace mace
