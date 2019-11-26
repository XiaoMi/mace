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

#include "mace/core/types.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class MVNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestMVNorm(const std::vector<index_t> &input_shape,
                const std::vector<T> &input,
                bool normalize_variance,
                bool across_channels,
                const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<D, T>(MakeString("Input"), input_shape, input);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  }
  OpDefBuilder("MVNorm", "MVNormTest")
      .Input(D == DeviceType::CPU ? "InputNCHW" : "Input")
      .AddIntArg("normalize_variance", normalize_variance)
      .AddIntArg("across_channels", across_channels)
      .Output(D == DeviceType::CPU ? "OutputNCHW" : "Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }

  net.AddInputFromArray<D, T>("ExpectedOutput", input_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(MVNormOpTest, SimpleTestMean) {
  TestMVNorm<DeviceType::CPU, float>(
    {1, 1, 5, 12},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    false, true,
    {-9.5, -8.5, -7.5, -6.5, -5.5,
     -4.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, -7.5, -6.5, -5.5,
     -4.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, -5.5,
     -4.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, 4.5,
     5.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, 4.5,
     5.5, 6.5, 7.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, 4.5,
     5.5, 6.5, 7.5, 8.5, 9.5});
}

TEST_F(MVNormOpTest, SimpleTestVariance) {
  TestMVNorm<DeviceType::CPU, float>(
    {1, 1, 5, 12},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    true, true,
    {-2.1287, -1.90463, -1.68056, -1.45648, -1.23241,
     -1.00833, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, -1.68056, -1.45648, -1.23241,
     -1.00833, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, -1.23241,
     -1.00833, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, 1.00833,
     1.23241, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, 1.00833,
     1.23241, 1.45648, 1.68056, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, 1.00833,
     1.23241, 1.45648, 1.68056, 1.90463, 2.1287});
}

TEST_F(MVNormOpTest, SimpleTestVariance2) {
  TestMVNorm<DeviceType::CPU, float>(
    {1, 1, 1, 16},
    {-0.63984936, -0.5024374 , -2.1083345,  2.6399455,
     -0.63989604, -0.63280314,  2.905462,  1.0263479,
     -0.502281  , -0.58158046, -0.5358325 , -0.50097936,
     1.2043145 , -0.53840625, -0.50652033, -0.48295242},
    true, true,
    {-0.485057, -0.376699, -1.643057, 2.101283,
     -0.485094, -0.479501, 2.310661, 0.828852,
     -0.376575, -0.439108, -0.403033, -0.375549,
     0.969191, -0.405062, -0.379918, -0.361333});
}

TEST_F(MVNormOpTest, SimpleTestMeanOpenCL) {
  TestMVNorm<DeviceType::GPU, float>(
    {1, 1, 5, 12},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    false, true,
    {-9.5, -8.5, -7.5, -6.5, -5.5,
     -4.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, -7.5, -6.5, -5.5,
     -4.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, -5.5,
     -4.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, 4.5,
     5.5, -3.5, -2.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, 4.5,
     5.5, 6.5, 7.5, -1.5, -0.5,
     0.5, 1.5, 2.5, 3.5, 4.5,
     5.5, 6.5, 7.5, 8.5, 9.5});
}


TEST_F(MVNormOpTest, SimpleTestVarianceOpenCL) {
  TestMVNorm<DeviceType::GPU, float>(
    {1, 1, 5, 12},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    true, true,
    {-2.1287, -1.90463, -1.68056, -1.45648, -1.23241,
     -1.00833, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, -1.68056, -1.45648, -1.23241,
     -1.00833, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, -1.23241,
     -1.00833, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, 1.00833,
     1.23241, -0.784259, -0.560185, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, 1.00833,
     1.23241, 1.45648, 1.68056, -0.336111, -0.112037,
     0.112037, 0.336111, 0.560185, 0.784259, 1.00833,
     1.23241, 1.45648, 1.68056, 1.90463, 2.1287});
}

namespace {
template <DeviceType D, typename T>
void TestMVNormRandom(const std::vector<index_t> &input_shape,
                      bool normalize_variance,
                      bool across_channels) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("MVNorm", "MVNormTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("normalize_variance", normalize_variance)
      .AddIntArg("across_channels", across_channels)
      .Finalize(net.NewOperatorDef());

  // run on cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  OpDefBuilder("MVNorm", "MVNormTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("normalize_variance", normalize_variance)
      .AddIntArg("across_channels", across_channels)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2, 1e-2);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  }
}
}  // namespace

TEST_F(MVNormOpTest, SimpleTestMeanHalfOpenCL) {
  TestMVNormRandom<DeviceType::GPU, half>({1, 1, 5, 12}, false, true);
}

TEST_F(MVNormOpTest, SimpleTestVarianceHalfOpenCL) {
  TestMVNormRandom<DeviceType::GPU, half>({1, 1, 5, 12}, true, true);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
