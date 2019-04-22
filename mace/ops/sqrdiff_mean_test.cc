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

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class SqrDiffMeanOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple(const std::vector<index_t> &input_shape0,
            const std::vector<float> &input0,
            const std::vector<index_t> &input_shape1,
            const std::vector<float> &input1,
            const std::vector<index_t> &output_shape,
            const std::vector<float> &output) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input0", input_shape0, input0);
  net.AddInputFromArray<D, float>("Input1", input_shape1, input1);

  net.TransformDataFormat<DeviceType::CPU, float>("Input0",
                                                  DataFormat::NHWC,
                                                  "InputNCHW0",
                                                  DataFormat::NCHW);
  net.TransformDataFormat<DeviceType::CPU, float>("Input1",
                                                  DataFormat::NHWC,
                                                  "InputNCHW1",
                                                  DataFormat::NCHW);

  if (D == DeviceType::CPU) {
    OpDefBuilder("SqrDiffMean", "SqrDiffMeanTest")
        .Input("InputNCHW0")
        .Input("InputNCHW1")
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW",
                                                    DataFormat::NCHW,
                                                    "Output",
                                                    DataFormat::NHWC);
  } else {
    OpDefBuilder("SqrDiffMean", "SqrDiffMeanTest")
        .Input("Input0")
        .Input("Input1")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }
  auto expected = net.CreateTensor<float>(output_shape, output);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5, 1e-3);
}

template <DeviceType D>
void Simple12Test() {
  Simple<D>({2, 2, 3, 4},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {2, 1, 1, 4},
            {1, 1, 1, 1, 1, 1, 1, 1},
            {2, 1, 1, 4},
            {127.667, 146.667, 167.667, 190.667,
             127.667, 146.667, 167.667, 190.667});
}


}  // namespace

TEST_F(SqrDiffMeanOpTest, CPUSimple12) {
  Simple12Test<DeviceType::CPU>();
}

TEST_F(SqrDiffMeanOpTest, GPUSimple12) {
  Simple12Test<DeviceType::GPU>();
}

namespace {
template <DeviceType D, typename T>
void RandomTest(const std::vector<index_t> &input_shape0,
                const std::vector<index_t> &input_shape1) {
  testing::internal::LogToStderr();
  srand(time(NULL));
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input0", input_shape0);
  net.AddRandomInput<D, float>("Input1", input_shape1);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input0", DataFormat::NHWC, "InputNCHW0", DataFormat::NCHW);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input1", DataFormat::NHWC, "InputNCHW1", DataFormat::NCHW);
  OpDefBuilder("SqrDiffMean", "SqrDiffMeanTest")
      .Input("InputNCHW0")
      .Input("InputNCHW1")
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  OpDefBuilder("SqrDiffMean", "SqrDiffMeanTest")
      .Input("Input0")
      .Input("Input1")
      .Output("OPENCLOutput")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);
  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-4, 1e-3);
  } else {
    ExpectTensorNear<float>(*net.GetTensor("Output"),
                            *net.GetOutput("OPENCLOutput"), 1e-2, 1e-2);
  }
}
}  // namespace

TEST_F(SqrDiffMeanOpTest, GPURandomFloat) {
  RandomTest<DeviceType::GPU, float>({4, 64, 64, 3}, {4, 1, 1, 3});
  RandomTest<DeviceType::GPU, float>({2, 64, 64, 4}, {2, 1, 1, 4});
  RandomTest<DeviceType::GPU, float>({8, 128, 128, 64}, {8, 1, 1, 64});
  RandomTest<DeviceType::GPU, float>({1, 640, 480, 64}, {1, 1, 1, 64});
  RandomTest<DeviceType::GPU, float>({8, 117, 87, 33}, {8, 1, 1, 33});
  RandomTest<DeviceType::GPU, float>({1, 619, 450, 61}, {1, 1, 1, 61});
  RandomTest<DeviceType::GPU, float>({11, 511, 561, 1}, {11, 1, 1, 1});
}

TEST_F(SqrDiffMeanOpTest, GPURandomHalf) {
  RandomTest<DeviceType::GPU, half>({4, 64, 64, 3}, {4, 1, 1, 3});
  RandomTest<DeviceType::GPU, half>({2, 64, 64, 4}, {2, 1, 1, 4});
  RandomTest<DeviceType::GPU, half>({8, 128, 128, 64}, {8, 1, 1, 64});
  RandomTest<DeviceType::GPU, half>({1, 640, 480, 64}, {1, 1, 1, 64});
  RandomTest<DeviceType::GPU, half>({8, 117, 87, 33}, {8, 1, 1, 33});
  RandomTest<DeviceType::GPU, half>({1, 619, 450, 61}, {1, 1, 1, 61});
  RandomTest<DeviceType::GPU, half>({11, 511, 561, 1}, {11, 1, 1, 1});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
