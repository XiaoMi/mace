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

#include <vector>

#include "mace/ops/eltwise.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class EltwiseOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T, typename DstType>
void SimpleScalarScalar(const ops::EltwiseType type,
                        const T input,
                        const float x,
                        const DstType output) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>("Input", {}, {input});

  if (D == DeviceType::CPU) {
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("Input")
        .AddIntArg("T", DataTypeToEnum<T>::v())
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("scalar_input", x)
        .OutputType({ops::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  auto expected = net.CreateTensor<DstType>({}, {output});

  ExpectTensorNear<DstType>(*expected, *net.GetOutput("Output"), 1e-5);
}

template <DeviceType D, typename T, typename DstType>
void SimpleTensorScalar(const ops::EltwiseType type,
                        const std::vector<index_t> &shape,
                        const std::vector<T> &input,
                        const float x,
                        const std::vector<DstType> &output) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>("Input", shape, input);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, T>("Input", NHWC, "TInput", NCHW);
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("TInput")
        .AddIntArg("T", DataTypeToEnum<T>::v())
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("scalar_input", x)
        .AddIntArg("data_format", DataFormat::NCHW)
        .OutputType({ops::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
        .Output("TOutput")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, DstType>("TOutput", NCHW, "Output", NHWC);
  } else {
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("Input")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatArg("scalar_input", x)
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  auto expected = net.CreateTensor<DstType>(shape, output);

  ExpectTensorNear<DstType>(*expected, *net.GetOutput("Output"), 1e-5);
}

template <DeviceType D, typename T, typename DstType>
void SimpleTensorEltwise(const ops::EltwiseType type,
                         const std::vector<index_t> &shape0,
                         const std::vector<T> &input0,
                         const std::vector<index_t> &shape1,
                         const std::vector<T> &input1,
                         const std::vector<DstType> &output,
                         const std::vector<float> &coeff = {}) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>("Input0", shape0, input0);
  net.AddInputFromArray<D, T>("Input1", shape1, input1);

  if (D == DeviceType::CPU) {
    auto op_builder =
        OpDefBuilder("Eltwise", "EltwiseTest")
            .AddIntArg("T", DataTypeToEnum<T>::v())
            .AddIntArg("type", static_cast<int>(type))
            .AddFloatsArg("coeff", coeff)
            .AddIntArg("data_format", DataFormat::NCHW)
            .OutputType({ops::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
            .Output("TOutput");
    if (shape0.size() > 1) {
      net.TransformDataFormat<D, T>("Input0", NHWC, "TInput0", NCHW);
      op_builder.Input("TInput0");
    } else {
      op_builder.Input("Input0");
    }
    if (shape1.size() > 1) {
      net.TransformDataFormat<D, T>("Input1", NHWC, "TInput1", NCHW);
      op_builder.Input("TInput1");
    } else {
      op_builder.Input("Input1");
    }
    op_builder.Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, DstType>("TOutput", NCHW, "Output", NHWC);
  } else {
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("Input0")
        .Input("Input1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", coeff)
        .Output("Output")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  std::vector<index_t> output_shape = shape0;
  if (input0.size() < input1.size()) {
    output_shape = shape1;
  }
  auto expected = net.CreateTensor<DstType>(output_shape, output);

  ExpectTensorNear<DstType>(*expected, *net.GetOutput("Output"), 1e-5);
}

template <DeviceType D, typename T, typename DstType>
void TensorGeneralBroadcastEltwise(const ops::EltwiseType type,
                                   const std::vector<index_t> &shape0,
                                   const std::vector<T> &input0,
                                   const std::vector<index_t> &shape1,
                                   const std::vector<T> &input1,
                                   const std::vector<index_t> &output_shape,
                                   const std::vector<DstType> &output,
                                   const std::vector<float> &coeff = {}) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>("Input0", shape0, input0);
  net.AddInputFromArray<D, T>("Input1", shape1, input1);

  if (D == DeviceType::CPU) {
    auto op_builder =
        OpDefBuilder("Eltwise", "EltwiseTest")
            .AddIntArg("T", DataTypeToEnum<T>::v())
            .Input("Input0")
            .Input("Input1")
            .AddIntArg("type", static_cast<int>(type))
            .AddFloatsArg("coeff", coeff)
            .OutputType({ops::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
            .Output("Output");
    op_builder.Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else if (D == DeviceType::GPU) {
    auto op_builder =
        OpDefBuilder("Eltwise", "EltwiseTest")
            .AddIntArg("T", DataTypeToEnum<T>::v())
            .Input("Input0")
            .Input("Input1")
            .AddIntArg("type", static_cast<int>(type))
            .AddFloatsArg("coeff", coeff)
            .OutputType({ops::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
            .Output("Output");
    op_builder.Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  auto expected = net.CreateTensor<DstType>(output_shape, output);
  ExpectTensorNear<DstType>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(EltwiseOpTest, CPUSimpleScalarScalar) {
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, 1, 2, 3);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, 1, 2, -1);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, 1, 2, 2);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, 1, 2, 0.5);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, 1, 2, 1);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, 1, 2, 2);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::NEG, 1, 2, -1);
  SimpleScalarScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::ABS, -1, 3, 1);
  SimpleScalarScalar<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, 1, 3, 0);
  SimpleScalarScalar<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, 3, 3, 1);
}

TEST_F(EltwiseOpTest, CPUSimpleTensorScalar) {
  SimpleTensorScalar<DeviceType::CPU, float, float>(ops::EltwiseType::SUM,
                                                    {1, 1, 1, 1}, {1}, 1, {2});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 1,
      {0, 1, 2, 3, 4, 5});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 2,
      {2, 4, 6, 8, 10, 12});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 2, 3}, {2, 4, 6, 8, 10, 12}, 2,
      {1, 2, 3, 4, 5, 6});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 1,
      {1, 1, 1, 1, 1, 1});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 3,
      {3, 3, 3, 4, 5, 6});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::NEG, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 3,
      {-1, -2, -3, -4, -5, -6});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::ABS, {1, 1, 2, 3}, {-1, -2, -3, -4, -5, -6}, 3,
      {1, 2, 3, 4, 5, 6});
  SimpleTensorScalar<DeviceType::CPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 1,
      {0, 1, 4, 9, 16, 25});
  SimpleTensorScalar<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 3,
      {0, 0, 1, 0, 0, 0});
}

TEST_F(EltwiseOpTest, GPUSimpleTensorScalar) {
  SimpleTensorScalar<DeviceType::GPU, float, float>(ops::EltwiseType::SUM,
                                                    {1, 1, 1, 1}, {1}, 1, {2});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 1,
      {0, 1, 2, 3, 4, 5});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 2,
      {2, 4, 6, 8, 10, 12});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 2, 3}, {2, 4, 6, 8, 10, 12}, 2,
      {1, 2, 3, 4, 5, 6});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 1,
      {1, 1, 1, 1, 1, 1});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::MAX, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 3,
      {3, 3, 3, 4, 5, 6});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::NEG, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 3,
      {-1, -2, -3, -4, -5, -6});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::ABS, {1, 1, 2, 3}, {-1, -2, -3, -4, -5, -6}, 3,
      {1, 2, 3, 4, 5, 6});
  SimpleTensorScalar<DeviceType::GPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, 1,
      {0, 1, 4, 9, 16, 25});
}

TEST_F(EltwiseOpTest, CPUSimpleTensorVector) {
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 1, 3},
      {1, 2, 3}, {2, 4, 6, 5, 7, 9});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {0, 0, 0, 0, 0, 5, 5, 5, 5, 5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, -5, -5, -5, -5, -5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 1, 3}, {1, 2, 3}, {1, 2, 1, 3},
      {1, 2, 3, 4, 5, 6}, {1, 4, 9, 4, 10, 18});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 1, 1, 5}, {1, 1, 1, 1, 5}, {1, 2, 3, 4, 1, 6, 7, 8, 9, 2});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 1, 5}, {1, 1, 1, 2, 4}, {1, 2, 1, 5},
      {1, 1, 1, 2, 2, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 2, 1, 1, 1, 2, 4});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 1, 5}, {1, 2, 3, 4, 5},
      {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {0, 0, 0, 0, 0, 25, 25, 25, 25, 25});
  SimpleTensorEltwise<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 1, 3}, {1, 2, 3}, {1, 1, 1, 0, 0, 0});

  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {3},
      {1, 2, 3}, {2, 4, 6, 5, 7, 9});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {5}, {1, 2, 3, 4, 5}, {0, 0, 0, 0, 0, 5, 5, 5, 5, 5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, -5, -5, -5, -5, -5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, {3}, {1, 2, 3}, {1, 2, 1, 3},
      {1, 2, 3, 4, 5, 6}, {1, 4, 9, 4, 10, 18});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {5}, {1, 1, 1, 1, 5}, {1, 2, 3, 4, 1, 6, 7, 8, 9, 2});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {5}, {1, 1, 1, 2, 4}, {1, 2, 1, 5},
      {1, 1, 1, 2, 2, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 2, 1, 1, 1, 2, 4});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, {5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, 25, 25, 25, 25, 25});
  SimpleTensorEltwise<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {3},
      {1, 2, 3}, {1, 1, 1, 0, 0, 0});
}

TEST_F(EltwiseOpTest, GPUSimpleTensorVector) {
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 1, 3},
      {1, 2, 3}, {2, 4, 6, 5, 7, 9});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUB, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {0, 0, 0, 0, 0, 5, 5, 5, 5, 5});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, -5, -5, -5, -5, -5});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 1, 3}, {1, 2, 3}, {1, 2, 1, 3},
      {1, 2, 3, 4, 5, 6}, {1, 4, 9, 4, 10, 18});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::DIV, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 1, 1, 5}, {1, 1, 1, 1, 5}, {1, 2, 3, 4, 1, 6, 7, 8, 9, 2});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 1, 5}, {1, 1, 1, 2, 4}, {1, 2, 1, 5},
      {1, 1, 1, 2, 2, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 2, 1, 1, 1, 2, 4});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MAX, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 1, 5}, {1, 2, 3, 4, 5},
      {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {0, 0, 0, 0, 0, 25, 25, 25, 25, 25});
}

TEST_F(EltwiseOpTest, CPUSimpleTensorTensor) {
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 3},
      {1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 3},
      {1, 2, 3, 4, 5, 6}, {0.2, 0.4, 0.6, 0.8, 1, 1.2}, {0.1, 0.1});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 1, 1, 5},
      {1, 2, 3, 4, 5}, {0, 0, 0, 0, 0});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, {1, 2, 1, 3}, {1, 2, 3, 4, 5, 6},
      {1, 2, 1, 3}, {1, 2, 3, 4, 5, 6}, {1, 4, 9, 16, 25, 36});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {1, 2, 1, 3}, {1, 2, 3, 4, 5, 6}, {1, 2, 1, 3},
      {1, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, 1});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5},
      {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 2, 1, 5}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  SimpleTensorEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, 25, 25, 25, 25, 25});
  SimpleTensorEltwise<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, 1});
}
TEST_F(EltwiseOpTest, GPUSimpleTensorTensor) {
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 3},
      {1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 3},
      {1, 2, 3, 4, 5, 6}, {0.2, 0.4, 0.6, 0.8, 1, 1.2}, {0.1, 0.1});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 1, 5}, {1, 2, 3, 4, 5}, {1, 1, 1, 5},
      {1, 2, 3, 4, 5}, {0, 0, 0, 0, 0});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::PROD, {1, 2, 1, 3}, {1, 2, 3, 4, 5, 6},
      {1, 2, 1, 3}, {1, 2, 3, 4, 5, 6}, {1, 4, 9, 16, 25, 36});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::DIV, {1, 2, 1, 3}, {1, 2, 3, 4, 5, 6}, {1, 2, 1, 3},
      {1, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, 1});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MIN, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5},
      {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MAX, {1, 2, 1, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 2, 1, 5}, {1, 2, 3, 4, 5, 1, 2, 3, 4, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  SimpleTensorEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, {1, 2, 1, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, 25, 25, 25, 25, 25});
}

namespace {
template <typename T>
void RandomTensorScalar(const ops::EltwiseType type,
                        const std::vector<index_t> &shape) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input", shape, false, true, true);

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                  NCHW);
  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("TInput")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatArg("scalar_input", 0.1)
      .AddIntArg("data_format", DataFormat::NCHW)
      .Output("TOutput")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(DeviceType::CPU);
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("Input")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatArg("scalar_input", 0.1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2, 1e-2);
  }
}

template <typename T>
void RandomTensorEltwise(const ops::EltwiseType type,
                         const std::vector<index_t> &shape0,
                         const std::vector<index_t> &shape1,
                         const std::vector<float> &coeff = {}) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input0",
                                             shape0,
                                             false,
                                             true,
                                             true);
  net.AddRandomInput<DeviceType::GPU, float>("Input1",
                                             shape1,
                                             false,
                                             true,
                                             true);

  net.TransformDataFormat<DeviceType::CPU, float>("Input0", NHWC, "TInput0",
                                                  NCHW);
  net.TransformDataFormat<DeviceType::CPU, float>("Input1", NHWC, "TInput1",
                                                  NCHW);
  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("TInput0")
      .Input("TInput1")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatsArg("coeff", coeff)
      .AddIntArg("data_format", DataFormat::NCHW)
      .Output("TOutput")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::CPU);
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatsArg("coeff", coeff)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2, 1e-2);
  }
}

void QuantizedSum(const std::vector<index_t> &shape) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input0",
                                             shape,
                                             false,
                                             true,
                                             true);
  net.AddRandomInput<DeviceType::CPU, float>("Input1",
                                             shape,
                                             false,
                                             true,
                                             true);

  net.TransformDataFormat<DeviceType::CPU, float>("Input0", NHWC, "TInput0",
                                                  NCHW);
  net.TransformDataFormat<DeviceType::CPU, float>("Input1", NHWC, "TInput1",
                                                  NCHW);

  OpDefBuilder("Eltwise", "EltwiseTest")
      .Input("TInput0")
      .Input("TInput1")
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::SUM))
      .AddIntArg("data_format", DataFormat::NCHW)
      .Output("TOutput")
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::CPU);
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);

  OpDefBuilder("Quantize", "QuantizeInput0")
      .Input("Input0")
      .Output("QuantizedInput0")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeInput1")
      .Input("Input1")
      .Output("QuantizedInput1")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeOutput")
      .Input("Output")
      .Output("ExpectedQuantizedOutput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Eltwise", "QuantizeEltwiseTest")
      .Input("QuantizedInput0")
      .Input("QuantizedInput1")
      .Output("QuantizedOutput")
      .AddIntArg("type", static_cast<int>(ops::EltwiseType::SUM))
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());
  net.Setup(DeviceType::CPU);
  Tensor *eq_output = net.GetTensor("ExpectedQuantizedOutput");
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(eq_output->scale());
  q_output->SetZeroPoint(eq_output->zero_point());
  net.Run();

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}
}  // namespace

TEST_F(EltwiseOpTest, RandomTensorScalarFloat) {
  RandomTensorScalar<float>(ops::EltwiseType::SUM, {1, 32, 32, 16});
  RandomTensorScalar<float>(ops::EltwiseType::SUB, {3, 32, 32, 16});
  RandomTensorScalar<float>(ops::EltwiseType::PROD, {1, 31, 37, 17});
  RandomTensorScalar<float>(ops::EltwiseType::DIV, {3, 31, 37, 17});
  RandomTensorScalar<float>(ops::EltwiseType::MIN, {1, 32, 32, 16});
  RandomTensorScalar<float>(ops::EltwiseType::MAX, {3, 31, 37, 17});
  RandomTensorScalar<float>(ops::EltwiseType::NEG, {1, 32, 32, 32});
  RandomTensorScalar<float>(ops::EltwiseType::ABS, {3, 31, 37, 17});
  RandomTensorScalar<float>(ops::EltwiseType::SQR_DIFF, {3, 31, 37, 17});
}

TEST_F(EltwiseOpTest, RandomTensorScalarHalf) {
  RandomTensorScalar<half>(ops::EltwiseType::SUM, {1, 32, 32, 16});
  RandomTensorScalar<half>(ops::EltwiseType::SUB, {3, 32, 32, 16});
  RandomTensorScalar<half>(ops::EltwiseType::PROD, {1, 31, 37, 17});
  RandomTensorScalar<half>(ops::EltwiseType::DIV, {3, 31, 37, 17});
  RandomTensorScalar<half>(ops::EltwiseType::MIN, {1, 32, 32, 16});
  RandomTensorScalar<half>(ops::EltwiseType::MAX, {3, 31, 37, 17});
  RandomTensorScalar<half>(ops::EltwiseType::NEG, {1, 32, 32, 32});
  RandomTensorScalar<half>(ops::EltwiseType::ABS, {3, 31, 37, 17});
  RandomTensorScalar<half>(ops::EltwiseType::SQR_DIFF, {3, 31, 37, 17});
}

TEST_F(EltwiseOpTest, RandomTensorVecFloat) {
  RandomTensorEltwise<float>(ops::EltwiseType::SUM, {1, 32, 32, 16},
                             {1, 1, 1, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::SUB, {5, 32, 32, 16},
                             {5, 1, 1, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::SUB, {5, 32, 32, 16},
                             {1, 1, 1, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::SUB, {5, 1, 1, 16},
                             {5, 32, 32, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::PROD, {1, 31, 37, 17},
                             {1, 1, 1, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::PROD, {1, 1, 1, 17},
                             {1, 31, 37, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::DIV, {3, 1, 1, 17},
                             {3, 31, 37, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::MIN, {1, 1, 1, 16},
                             {1, 32, 32, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::MAX, {5, 31, 37, 17},
                             {5, 1, 1, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::SQR_DIFF, {5, 31, 37, 17},
                             {5, 1, 1, 17});
}

TEST_F(EltwiseOpTest, RandomTensorVecHalf) {
  RandomTensorEltwise<half>(ops::EltwiseType::SUM, {1, 32, 32, 16},
                            {1, 1, 1, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::SUB, {3, 32, 32, 16},
                            {3, 1, 1, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::SUB, {3, 32, 32, 16},
                            {1, 1, 1, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::SUB, {3, 1, 1, 16},
                            {3, 32, 32, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::PROD, {1, 1, 1, 17},
                            {1, 31, 37, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::DIV, {5, 31, 37, 17},
                            {5, 1, 1, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::DIV, {5, 31, 37, 17},
                            {1, 1, 1, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::DIV, {5, 1, 1, 17},
                            {5, 31, 37, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::MIN, {1, 1, 1, 16},
                            {1, 32, 32, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::MAX, {3, 31, 37, 17},
                            {3, 1, 1, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::SQR_DIFF, {3, 31, 37, 17},
                            {3, 1, 1, 17});
}

TEST_F(EltwiseOpTest, RandomTensorTensorFloat) {
  RandomTensorEltwise<float>(ops::EltwiseType::SUM, {1, 32, 32, 16},
                             {1, 32, 32, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::SUB, {3, 32, 32, 16},
                             {3, 32, 32, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::PROD, {1, 31, 37, 17},
                             {1, 31, 37, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::DIV, {5, 31, 37, 17},
                             {5, 31, 37, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::MIN, {1, 32, 32, 16},
                             {1, 32, 32, 16});
  RandomTensorEltwise<float>(ops::EltwiseType::MAX, {3, 31, 37, 17},
                             {3, 31, 37, 17});
  RandomTensorEltwise<float>(ops::EltwiseType::SQR_DIFF, {3, 31, 37, 17},
                             {3, 31, 37, 17});
}

TEST_F(EltwiseOpTest, RandomTensorTensorHalf) {
  RandomTensorEltwise<half>(ops::EltwiseType::SUM, {1, 32, 32, 16},
                            {1, 32, 32, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::SUB, {3, 32, 32, 16},
                            {3, 32, 32, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::PROD, {1, 31, 37, 17},
                            {1, 31, 37, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::DIV, {5, 31, 37, 17},
                            {5, 31, 37, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::MIN, {1, 32, 32, 16},
                            {1, 32, 32, 16});
  RandomTensorEltwise<half>(ops::EltwiseType::MAX, {3, 31, 37, 17},
                            {3, 31, 37, 17});
  RandomTensorEltwise<half>(ops::EltwiseType::SQR_DIFF, {3, 31, 37, 17},
                            {3, 31, 37, 17});
}

TEST_F(EltwiseOpTest, TensorGeneralBroadcastCPU) {
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {2, 3, 4, 6, 7, 8});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {0, 1, 2, 2, 3, 4});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 3}, {1, 2, 3, 8, 10, 12});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {1, 2, 3, 2, 2.5, 3});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {1, 1, 1, 2, 2, 2});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 3}, {0, 1, 4, 4, 9, 16});
  TensorGeneralBroadcastEltwise<DeviceType::CPU, int32_t, int32_t>(
      ops::EltwiseType::EQUAL, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 3}, {1, 0, 0, 0, 0, 0});
}

TEST_F(EltwiseOpTest, TensorGeneralBroadcastGPU) {
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {2, 3, 4, 6, 7, 8});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {0, 1, 2, 2, 3, 4});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 3}, {1, 2, 3, 8, 10, 12});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {1, 2, 3, 2, 2.5, 3});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {1, 1, 1, 2, 2, 2});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MAX, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1, 2, 1},
      {1, 2}, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 3}, {0, 1, 4, 4, 9, 16});

  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUM, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 5}, {1, 2, 3, 4, 5, 7, 8, 9, 10, 11});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SUB, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 5}, {-1, 0, 1, 2, 3, 3, 4, 5, 6, 7});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::PROD, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 10, 12, 14, 16, 18});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::DIV, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {1, 2}, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 2.5, 3, 3.5, 4, 4.5});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MIN, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {3, 4}, {1, 1, 2, 5}, {0, 1, 2, 3, 3, 4, 4, 4, 4, 4});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::MAX, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {3, 4}, {1, 1, 2, 5}, {3, 3, 3, 3, 4, 5, 6, 7, 8, 9});
  TensorGeneralBroadcastEltwise<DeviceType::GPU, float, float>(
      ops::EltwiseType::SQR_DIFF, {1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {1, 1, 2, 1}, {2, 3}, {1, 1, 2, 5}, {4, 1, 0, 1, 4, 4, 9, 16, 25, 36});
}

TEST_F(EltwiseOpTest, QuantizedSum) {
  QuantizedSum({1, 32, 32, 16});
  QuantizedSum({1, 31, 31, 17});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
