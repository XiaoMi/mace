//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/hexagon_control_wrapper.h"
#include "mace/dsp/util/quantize.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/pooling.h"
#include "gtest/gtest.h"

using namespace mace;

static NetDef BuildNetDef(const vector<index_t> &input_shape,
                          const vector<index_t> &output_shape,
                          const vector<index_t> &filter_shape,
                          const vector<int> &stride,
                          Padding padding,
                          float input_min, float input_max) {
  NetDef net;
  net.set_name("quantized_maxpool_test");
  // input op
  OperatorDef *input_op = net.add_op();
  input_op->set_name("input_node");
  input_op->set_type("INPUT");
  input_op->set_node_id(0);
  input_op->set_padding(0);

  // maxpool op
  OperatorDef *maxpool_op = net.add_op();
  maxpool_op->set_name("maxpool");
  maxpool_op->set_type("QuantizedMaxPool_8");
  maxpool_op->set_node_id(1);
  if (padding == Padding::SAME) {
    maxpool_op->set_padding(1);
  } else {
    maxpool_op->set_padding(2);
  }
  maxpool_op->add_input("input_node");
  maxpool_op->add_input("input_min");
  maxpool_op->add_input("input_max");
  maxpool_op->add_input("ksize");
  maxpool_op->add_input("stride");
  maxpool_op->add_output("maxpool:0");
  maxpool_op->add_output("maxpool:1");
  maxpool_op->add_output("maxpool:2");
  NodeInput *input_node_input = maxpool_op->add_node_input();
  input_node_input->set_node_id(0);
  input_node_input->set_output_port(0);
  input_node_input = maxpool_op->add_node_input();
  input_node_input->set_node_id(10);
  input_node_input->set_output_port(0);
  input_node_input = maxpool_op->add_node_input();
  input_node_input->set_node_id(11);
  input_node_input = maxpool_op->add_node_input();
  input_node_input->set_node_id(12);
  input_node_input->set_output_port(0);
  input_node_input = maxpool_op->add_node_input();
  input_node_input->set_node_id(13);
  input_node_input->set_output_port(0);

  // output op
  OperatorDef *output_op = net.add_op();
  output_op->set_name("__output__");
  output_op->set_type("OUTPUT");
  output_op->set_op_id(2);
  input_node_input = output_op->add_node_input();
  input_node_input->set_node_id(1);
  input_node_input->set_output_port(0);

  // tensor
  TensorProto *input_min_tensor = net.add_tensors();
  input_min_tensor->set_name("input_min");
  input_min_tensor->add_dims(1);
  input_min_tensor->set_data_type(DataType::DT_FLOAT);
  input_min_tensor->set_node_id(10);
  input_min_tensor->add_float_data(input_min);

  TensorProto *input_max_tensor = net.add_tensors();
  input_max_tensor->set_name("input_max");
  input_max_tensor->add_dims(1);
  input_max_tensor->set_data_type(DataType::DT_FLOAT);
  input_max_tensor->set_node_id(11);
  input_max_tensor->add_float_data(input_max);

  TensorProto *ksize_tensor = net.add_tensors();
  ksize_tensor->set_name("ksize");
  ksize_tensor->add_dims(filter_shape[0]);
  ksize_tensor->add_dims(filter_shape[1]);
  ksize_tensor->add_dims(filter_shape[2]);
  ksize_tensor->add_dims(filter_shape[3]);
  ksize_tensor->set_data_type(DataType::DT_INT32);
  ksize_tensor->set_node_id(12);

  TensorProto *stride_tensor = net.add_tensors();
  stride_tensor->set_name("stride");
  stride_tensor->add_dims(stride[0]);
  stride_tensor->add_dims(stride[1]);
  stride_tensor->add_dims(stride[2]);
  stride_tensor->add_dims(stride[3]);
  stride_tensor->set_data_type(DataType::DT_INT32);
  stride_tensor->set_node_id(13);

  // input & output info
  InputInfo *input_info = net.add_input_info();
  input_info->set_name("input_node");
  input_info->set_node_id(0);
  input_info->add_dims(input_shape[0]);
  input_info->add_dims(input_shape[1]);
  input_info->add_dims(input_shape[2]);
  input_info->add_dims(input_shape[3]);
  input_info->set_data_type(DataType::DT_UINT8);
  input_info->set_max_byte_size(1000);
  OutputInfo *output_info = net.add_output_info();
  output_info->set_name("output_node");
  output_info->set_node_id(1);
  output_info->add_dims(output_shape[0]);
  output_info->add_dims(output_shape[1]);
  output_info->add_dims(output_shape[2]);
  output_info->add_dims(output_shape[3]);
  output_info->set_data_type(DataType::DT_UINT8);
  output_info->set_max_byte_size(1000);

  return net;
}

static void TestQuantizedMaxPool(Padding padding, int kernel_size, int stride_size) {
  testing::internal::LogToStderr();
  HexagonControlWrapper wrapper;
  wrapper.Init();
  wrapper.SetDebugLevel(3);
  wrapper.Config();

  vector<index_t> input_shape {1, 10, 10, 3};
  vector<index_t> filter_shape {1, 3, 3, 1};
  vector<int> stride {1, stride_size, stride_size, 1};
  vector<int> dilation {1, 1, 1, 1};
  vector<index_t> output_shape {input_shape[0], 0, 0, input_shape[3]};
  vector<int> padding_size(2);
  switch (padding) {
    case VALID:
      output_shape[1] = (input_shape[1] - filter_shape[1]) / stride[1] + 1;
      output_shape[2] = (input_shape[2] - filter_shape[2]) / stride[2] + 1;
      break;
    case SAME:
      output_shape[1] = (input_shape[1] - 1) / stride[1] + 1;
      output_shape[2] = (input_shape[2] - 1) / stride[2] + 1;
      break;
    default:
      ASSERT_TRUE(0);
  }
  for (int i = 0; i < 4; ++i) {
    VLOG(0) << "! shape = " << output_shape[i];
  }
  NetDef net = BuildNetDef(input_shape, output_shape, filter_shape, stride,
                           padding, -50, 100);
  VLOG(0) << wrapper.SetupGraph(net);

  Allocator *cpu_allocator = GetDeviceAllocator(DeviceType::CPU);
  Tensor original_tensor(cpu_allocator, DT_FLOAT);
  Tensor input_tensor(cpu_allocator, DT_UINT8);
  Tensor output_tensor(cpu_allocator, DT_UINT8);
  Tensor dequantized_output_tensor(cpu_allocator, DT_FLOAT);
  original_tensor.Resize(input_shape);
  input_tensor.Resize(input_shape);
  output_tensor.Resize(output_shape);
  dequantized_output_tensor.Resize(output_shape);
  float *original_data = original_tensor.mutable_data<float>();
  uint8_t *input_data = input_tensor.mutable_data<uint8_t>();
  const uint8_t *output_data = output_tensor.data<uint8_t>();
  float *dequantized_output_data = dequantized_output_tensor.mutable_data<float>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(-50, 50);
  std::generate(original_data, original_data + original_tensor.size(),
                [&gen, &nd] {
                  return nd(gen);
                });

  Quantizer quantizer;
  float min_in, min_out;
  quantizer.Quantize(original_tensor, &input_tensor, &min_in, &min_out);
  VLOG(0) << wrapper.ExecuteGraph(input_tensor, &output_tensor);
  quantizer.DeQuantize(output_tensor, min_in, min_out, &dequantized_output_tensor);

  // debug original float input data
  for (index_t c = 0; c < input_shape[3]; ++c) {
    for (index_t i = 0; i < input_shape[1]; ++i) {
      for (index_t j = 0; j < input_shape[2]; ++j) {
        std::cout << original_data[i * input_shape[2] * input_shape[3] + j * input_shape[3] + c] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }

  // debug dequantized float output data
  for (index_t c = 0; c < output_shape[3]; ++c) {
    for (index_t i = 0; i < output_shape[1]; ++i) {
      for (index_t j = 0; j < output_shape[2]; ++j) {
        std::cout << dequantized_output_data[i * output_shape[2] * output_shape[3] + j * output_shape[3] + c] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }

  wrapper.PrintLog();
  VLOG(0) << wrapper.TeardownGraph();
  wrapper.Finalize();
}

TEST(QuantizedMaxPoolTest, QuantizedMaxPoolValidStride1) {
  TestQuantizedMaxPool(Padding::VALID, 3, 1);
}

TEST(QuantizedMaxPoolTest, QuantizedMaxPoolValidStride2) {
  TestQuantizedMaxPool(Padding::VALID, 3, 2);
}

TEST(QuantizedMaxPoolTest, QuantizedMaxPoolSameStride1) {
  TestQuantizedMaxPool(Padding::SAME, 3, 1);
}

TEST(QuantizedMaxPoolTest, QuantizedMaxPoolSameStride2) {
  TestQuantizedMaxPool(Padding::SAME, 3, 2);
}