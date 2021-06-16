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

namespace mace {
namespace ops {
namespace test {

class InstanceNormOpTest : public OpsTestBase {};

namespace {
template <RuntimeType D>
void SimpleAffineTrue() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 3, 2, 2},
      {3.8758793, 9.386173, 2.3559103, -3.3926811, 0.95884323, -1.1207237,
       4.0818605, 1.1450119, 3.9172459, 9.698574, -4.152527, -0.35342026,
       2.2998524, -0.06531906, -0.9577513, -8.849655, -8.62608, -8.998599,
       -9.784871, -9.313494, -7.575619, -9.019365, -9.380671, 4.38369});
  net.AddInputFromArray<D, float>(
      "Scale",
      {3},
      {0.22904232144355774, 0.24142828583717346, 0.15624916553497314},
      true);
  net.AddInputFromArray<D, float>(
      "Offset",
      {3},
      {0.3075108230113983, 0.10697364807128906, 0.20591309666633606},
      true);

  if (D == RuntimeType::RT_CPU) {
    OpDefBuilder("InstanceNorm", "InstanceNormTest")
        .Input("Input")
        .Input("Scale")
        .Input("Offset")
        .AddFloatArg("epsilon", 0.000009999999747378752)
        .AddIntArg("affine", 1)
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  } else if (D == RuntimeType::RT_OPENCL) {
    net.TransformDataFormat<D, float>(
        "Input", DataFormat::NCHW, "InputNHWC", DataFormat::NHWC);
    OpDefBuilder("InstanceNorm", "InstanceNormTest")
        .Input("InputNHWC")
        .Input("Scale")
        .Input("Offset")
        .AddFloatArg("epsilon", 0.000009999999747378752)
        .AddIntArg("affine", 1)
        .Output("OutputNHWC")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>(
        "OutputNHWC", DataFormat::NHWC, "Output", DataFormat::NCHW);
  }

  // take pytorch's output as ground truth
  auto expected = net.CreateTensor<float>(
      {2, 3, 2, 2},
      {0.34876436, 0.6261319, 0.2722548, -0.017107755, 0.06692189, -0.20402485,
       0.4738198, 0.0911778, 0.25567788, 0.4311325, 0.010772422, 0.12606959,
       0.53677684, 0.4074555, 0.35865968, -0.07284871, 0.42186037, 0.21038517,
       -0.23597279, 0.03162293, 0.14609051, 0.106428735, 0.09650315,
       0.47462997});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4);
}

template <RuntimeType D>
void SimpleAffineFalse() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {2, 3, 2, 2},
      {-0.8383074, -0.3428688, -3.7500334, 2.3004332, -5.721054, -1.7634764,
       3.8758793, 9.386173, 2.3559103, -3.3926811, 0.95884323, -1.1207237,
       4.0818605, 1.1450119, 3.9172459, 9.698574, -4.152527, -0.35342026,
       2.2998524, -0.06531906, -0.9577513, -8.849655, -8.62608, -8.998599});

  if (D == RuntimeType::RT_CPU) {
    OpDefBuilder("InstanceNorm", "InstanceNormTest")
        .Input("Input")
        .AddFloatArg("epsilon", 0.000009999999747378752)
        .AddIntArg("affine", 0)
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  } else if (D == RuntimeType::RT_OPENCL) {
    net.TransformDataFormat<D, float>(
        "Input", DataFormat::NCHW, "InputNHWC", DataFormat::NHWC);
    OpDefBuilder("InstanceNorm", "InstanceNormTest")
        .Input("InputNHWC")
        .AddFloatArg("epsilon", 0.000009999999747378752)
        .AddIntArg("affine", 0)
        .Output("OutputNHWC")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>(
        "OutputNHWC", DataFormat::NHWC, "Output", DataFormat::NCHW);
  }

  // take pytorch's output as ground truth
  auto expected = net.CreateTensor<float>(
      {2, 3, 2, 2},
      {-0.08410892, 0.14660946, -1.4400564, 1.3775558, -1.2539229, -0.5613624,
       0.42550275, 1.3897825, 1.2224286, -1.4237958, 0.5793227, -0.37795544,
       -0.20237465, -1.1475586, -0.2553536, 1.6052865, -1.5507977, 0.09276787,
       1.240624, 0.21740592, 1.7307396, -0.5842105, -0.5186286, -0.62790054});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4);
}
}  // namespace

TEST_F(InstanceNormOpTest, SimpleAffineTrueCPU) {
  SimpleAffineTrue<RuntimeType::RT_CPU>();
}

TEST_F(InstanceNormOpTest, SimpleAffineTrueOPENCL) {
  SimpleAffineTrue<RuntimeType::RT_OPENCL>();
}

TEST_F(InstanceNormOpTest, SimpleAffineFalseCPU) {
  SimpleAffineFalse<RuntimeType::RT_CPU>();
}

TEST_F(InstanceNormOpTest, SimpleAffineFalseOPENCL) {
  SimpleAffineFalse<RuntimeType::RT_OPENCL>();
}

TEST_F(InstanceNormOpTest, SimpleAffineTrueHalfOPENCL) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<RuntimeType::RT_OPENCL, float>(
      "Input", {2, 3, 2, 2},
      {3.8758793, 9.386173, 2.3559103, -3.3926811, 0.95884323, -1.1207237,
       4.0818605, 1.1450119, 3.9172459, 9.698574, -4.152527, -0.35342026,
       2.2998524, -0.06531906, -0.9577513, -8.849655, -8.62608, -8.998599,
       -9.784871, -9.313494, -7.575619, -9.019365, -9.380671, 4.38369});
  net.AddInputFromArray<RuntimeType::RT_OPENCL, float>(
      "Scale",
      {3},
      {0.22904232144355774, 0.24142828583717346, 0.15624916553497314},
      true);
  net.AddInputFromArray<RuntimeType::RT_OPENCL, float>(
      "Offset",
      {3},
      {0.3075108230113983, 0.10697364807128906, 0.20591309666633606},
      true);

  net.TransformDataFormat<RuntimeType::RT_OPENCL, float>(
      "Input", DataFormat::NCHW, "InputNHWC", DataFormat::NHWC);
  OpDefBuilder("InstanceNorm", "InstanceNormTest")
      .Input("InputNHWC")
      .Input("Scale")
      .Input("Offset")
      .AddFloatArg("epsilon", 0.000009999999747378752)
      .AddIntArg("affine", 1)
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Output("OutputNHWC")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(RuntimeType::RT_OPENCL);
  net.TransformDataFormat<RuntimeType::RT_OPENCL, float>(
      "OutputNHWC", DataFormat::NHWC, "Output", DataFormat::NCHW);

  // take pytorch's output as ground truth
  auto expected = net.CreateTensor<float>(
      {2, 3, 2, 2},
      {0.34876436, 0.6261319, 0.2722548, -0.017107755, 0.06692189, -0.20402485,
       0.4738198, 0.0911778, 0.25567788, 0.4311325, 0.010772422, 0.12606959,
       0.53677684, 0.4074555, 0.35865968, -0.07284871, 0.42186037, 0.21038517,
       -0.23597279, 0.03162293, 0.14609051, 0.106428735, 0.09650315,
       0.47462997});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-1, 1e-2);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
