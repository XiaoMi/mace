//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/util/quantize.h"
#include "gtest/gtest.h"

using namespace mace;

TEST(QuantizeTest, QuantizeAndDequantize) {
  testing::internal::LogToStderr();

  Quantizer quantizer;
  Allocator *allocator = GetDeviceAllocator(DeviceType::CPU);

  Tensor in_tensor(allocator, DataType::DT_FLOAT);
  vector<index_t> shape {5};
  in_tensor.Resize(shape);
  float *in_data = in_tensor.mutable_data<float>();
  in_data[0] = -50.0;
  in_data[1] = -10.0;
  in_data[2] = 20.0;
  in_data[3] = 80.0;
  in_data[4] = 100.0;

  Tensor quantized_tensor(allocator, DataType::DT_UINT8);
  quantized_tensor.Resize(shape);
  uint8_t *quantized_data = quantized_tensor.mutable_data<uint8_t>();
  float min_out, max_out;
  quantizer.Quantize(in_tensor, -50.0, 100.0, &quantized_tensor, &min_out, &max_out);
  vector<uint8_t> expected_quantize_data {0, 68, 119, 220, 254};
  for (int i = 0; i < quantized_tensor.size(); ++i) {
    EXPECT_EQ(expected_quantize_data[i], quantized_data[i]);
  }

  Tensor dequantized_tensor(allocator, DataType::DT_FLOAT);
  dequantized_tensor.Resize(shape);
  float *dequantized_data = dequantized_tensor.mutable_data<float>();
  quantizer.DeQuantize(quantized_tensor, min_out, max_out, &dequantized_tensor);

  for (int i = 0; i < dequantized_tensor.size(); ++i) {
    EXPECT_NEAR(in_data[i], dequantized_data[i], 1);
  }
}

