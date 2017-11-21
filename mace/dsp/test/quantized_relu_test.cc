//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/hexagon_control_wrapper.h"
#include "gtest/gtest.h"

using namespace mace;

static NetDef BuildNetDef() {
  NetDef net;
  net.set_name("quantized_relu_test");
  // input op
  OperatorDef *input_op = net.add_op();
  input_op->set_name("input_node");
  input_op->set_type("INPUT");
  input_op->set_node_id(0);
  input_op->set_padding(0);
  input_op->add_out_max_byte_size(1000);

  // relu op
  OperatorDef *relu_op = net.add_op();
  relu_op->set_name("relu");
  relu_op->set_type("QuantizedRelu_8");
  relu_op->set_node_id(1);
  relu_op->set_padding(0);
  relu_op->add_input("input_node");
  relu_op->add_input("input_min");
  relu_op->add_input("input_max");
  relu_op->add_output("relu:0");
  relu_op->add_output("relu:1");
  relu_op->add_output("relu:2");
  NodeInput *input_node_input = relu_op->add_node_input();
  input_node_input->set_node_id(0);
  input_node_input->set_output_port(0);
  input_node_input = relu_op->add_node_input();
  input_node_input->set_node_id(10);
  input_node_input->set_output_port(0);
  input_node_input = relu_op->add_node_input();
  input_node_input->set_node_id(11);
  input_node_input->set_output_port(0);
  relu_op->add_out_max_byte_size(1000);
  relu_op->add_out_max_byte_size(1000);
  relu_op->add_out_max_byte_size(1000);

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
  input_min_tensor->add_float_data(-100.0);

  TensorProto *input_max_tensor = net.add_tensors();
  input_max_tensor->set_name("input_max");
  input_max_tensor->add_dims(1);
  input_max_tensor->set_data_type(DataType::DT_FLOAT);
  input_max_tensor->set_node_id(11);
  input_max_tensor->add_float_data(100.0);

  // input & output info
  InputInfo *input_info = net.add_input_info();
  input_info->set_name("input_node");
  input_info->set_node_id(0);
  input_info->add_dims(1);
  input_info->add_dims(1);
  input_info->add_dims(1);
  input_info->add_dims(5);
  input_info->set_data_type(DataType::DT_UINT8);
  input_info->set_max_byte_size(1000);
  OutputInfo *output_info = net.add_output_info();
  output_info->set_name("output_node");
  output_info->set_node_id(1);
  output_info->add_dims(1);
  output_info->add_dims(1);
  output_info->add_dims(1);
  output_info->add_dims(5);
  output_info->set_data_type(DataType::DT_UINT8);
  output_info->set_max_byte_size(1000);

  return net;
}

TEST(QuantizedReluTest, QuantizedRelu) {
  testing::internal::LogToStderr();
  HexagonControlWrapper wrapper;
  wrapper.Init();
  wrapper.SetDebugLevel(3);
  wrapper.Config();

  NetDef net = BuildNetDef();
  wrapper.SetupGraph(net);

  Allocator *cpu_allocator = GetDeviceAllocator(DeviceType::CPU);
  Tensor input_tensor(cpu_allocator, DT_UINT8);
  Tensor output_tensor(cpu_allocator, DT_UINT8);
  input_tensor.Resize({1, 1, 1, 5});
  output_tensor.Resize({1, 1, 1, 5});
  uint8_t *input_data = input_tensor.mutable_data<uint8_t>();
  const uint8_t *output_data = output_tensor.data<uint8_t>();
  // -100.0 -50.0 0 50.0 100.0 -> s=0.782, q0=int(-fmin/s)=128,
  // q=q0+f/s -> 0, 64, 128, 192, 256
  input_data[0] = 0;
  input_data[1] = 64;
  input_data[2] = 128;
  input_data[3] = 192;
  input_data[4] = 255;

  // 0, 0, 0, 50, 100 -> s=0.782, q0=128
  // q -> 128, 128, 128, 192, 255

  VLOG(0) << wrapper.ExecuteGraph(input_tensor, &output_tensor);
  wrapper.PrintLog();

  for (int i = 0; i < output_tensor.size(); ++i) {
    std::cout << (int32_t) output_data[i] << " ";
  }
  std::cout << std::endl;

  VLOG(0) << wrapper.TeardownGraph();
  wrapper.Finalize();
}