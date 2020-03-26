// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/core/quantize.h"
#include "mace/core/runtime/hexagon/hexagon_hta_transformer.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class HTATransformTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void TestHTAQuantizeDequantize(const std::vector<float> &input) {
  float min_val, max_val;
  FindMinMax(input.data(), input.size(), &min_val, &max_val);
  float scale;
  int32_t zero;
  AdjustRange<uint8_t>(min_val, max_val, false, &scale, &zero);

  OpsTestNet net;
    Device *device = OpTestContext::Get()->GetDevice(D);

  net.AddInputFromArray<D, float>("Input",
                                  {static_cast<index_t>(input.size())},
                                  input);
  Tensor *input_tensor = net.GetOutput("Input");
  input_tensor->SetScale(scale);
  input_tensor->SetZeroPoint(zero);
  Tensor *quantized_output = net.ws()->CreateTensor(
      "QuantizedOutput", device->allocator(), DT_UINT8);
  Tensor *dequantized_output = net.ws()->CreateTensor(
      "DequantizedOutput", device->allocator(), DT_FLOAT);

  mace::HexagonHTATranformer<D> transformer;
  transformer.Init(device);
  transformer.Quantize(input_tensor, quantized_output);
  transformer.Dequantize(quantized_output, dequantized_output);

  ExpectTensorNear<float>(*input_tensor,
                          *dequantized_output,
                          0.1);
}

}  // namespace

TEST_F(HTATransformTest, TestHTAQuantize) {
  TestHTAQuantizeDequantize<CPU>({-2, -1, 0, 1, 2, 3, 4});
  TestHTAQuantizeDequantize<GPU>({-2, -1, 0, 1, 2, 3, 4});
}

namespace {
void TestHTAInputTransform(const std::vector<index_t> &input_shape,
                           const hexagon_hta_hw_layout format) {
  OpsTestNet net;
  Device *device = OpTestContext::Get()->GetDevice(DeviceType::GPU);
  net.AddRandomInput<GPU, uint8_t>("Input", input_shape);
  Tensor *input_tensor = net.GetOutput("Input");
  input_tensor->SetScale(0.1);
  input_tensor->SetZeroPoint(1);
  Tensor *cpu_transformed_tensor = net.ws()->CreateTensor(
      "CpuTransformedOutput", device->allocator(), DT_UINT8);
  Tensor *gpu_transformed_tensor = net.ws()->CreateTensor(
      "GpuTransformedOutput", device->allocator(), DT_UINT8);

  mace::HexagonHTATranformer<CPU> cpu_transformer;
  mace::HexagonHTATranformer<GPU> gpu_transformer;
  cpu_transformer.Init(device);
  gpu_transformer.Init(device);
  cpu_transformer.SetInputTransformer(format);
  gpu_transformer.SetInputTransformer(format);
  cpu_transformer.TransformInput(input_tensor, cpu_transformed_tensor, 0);
  gpu_transformer.TransformInput(input_tensor, gpu_transformed_tensor, 0);

  net.Sync();
  ExpectTensorNear<uint8_t>(*cpu_transformed_tensor, *gpu_transformed_tensor);
}

}  // namespace

TEST_F(HTATransformTest, TestHTAInputTransform) {
  TestHTAInputTransform({1, 15, 33, 2}, HEXAGON_HTA_HW_FORMAT_PLANAR);
  TestHTAInputTransform({1, 19, 31, 3}, HEXAGON_HTA_HW_FORMAT_PLANAR);
  TestHTAInputTransform({1, 224, 224, 3}, HEXAGON_HTA_HW_FORMAT_PLANAR);
  TestHTAInputTransform({1, 19, 31, 3}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAInputTransform({1, 15, 33, 27}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAInputTransform({1, 15, 33, 35}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAInputTransform({1, 224, 224, 3}, HEXAGON_HTA_HW_FORMAT_D32);
}


namespace {
void TestHTAOutputTransform(const std::vector<index_t> &output_shape,
                            const hexagon_hta_hw_layout format) {
  index_t batch = output_shape[0];
  index_t height = output_shape[1];
  index_t width = output_shape[2];
  index_t channels = output_shape[3];
  MACE_CHECK(format == HEXAGON_HTA_HW_FORMAT_D32);
  std::vector<index_t> input_shape {
      batch, height, RoundUpDiv<index_t>(channels, 32), width, 32};

  OpsTestNet net;
  Device *device = OpTestContext::Get()->GetDevice(DeviceType::GPU);
  net.AddRandomInput<GPU, uint8_t>("Input", input_shape);
  Tensor *input_tensor = net.GetOutput("Input");
  Tensor *cpu_transformed_tensor = net.ws()->CreateTensor(
      "CpuTransformedOutput", device->allocator(), DT_UINT8);
  Tensor *gpu_transformed_tensor = net.ws()->CreateTensor(
      "GpuTransformedOutput", device->allocator(), DT_UINT8);
  cpu_transformed_tensor->Resize(output_shape);
  gpu_transformed_tensor->Resize(output_shape);

  mace::HexagonHTATranformer<CPU> cpu_transformer;
  mace::HexagonHTATranformer<GPU> gpu_transformer;
  cpu_transformer.Init(device);
  gpu_transformer.Init(device);
  cpu_transformer.SetOutputTransformer(format);
  gpu_transformer.SetOutputTransformer(format);
  cpu_transformer.TransformOutput(input_tensor, cpu_transformed_tensor, 0);
  gpu_transformer.TransformOutput(input_tensor, gpu_transformed_tensor, 0);

  net.Sync();
  ExpectTensorNear<uint8_t>(*cpu_transformed_tensor, *gpu_transformed_tensor);
}

}  // namespace

TEST_F(HTATransformTest, TestHTAOutputTransform) {
  TestHTAOutputTransform({1, 15, 33, 2}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAOutputTransform({1, 19, 31, 27}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAOutputTransform({1, 19, 31, 35}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAOutputTransform({1, 224, 224, 2}, HEXAGON_HTA_HW_FORMAT_D32);
  TestHTAOutputTransform({1, 384, 384, 3}, HEXAGON_HTA_HW_FORMAT_D32);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
