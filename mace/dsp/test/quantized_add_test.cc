//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/hexagon_control_wrapper.h"
#include "gtest/gtest.h"
#include <math.h>

using namespace mace;

static NetDef BuildNetDef() {
  NetDef net;
  net.set_name("quantized_add_test");
  // input op
  OperatorDef *input_op = net.add_op();
  input_op->set_name("input_node");
  input_op->set_type("INPUT");
  input_op->set_node_id(0);
  input_op->set_padding(0);
  input_op->add_out_max_byte_size(1000);

  // add op
  OperatorDef *add_op = net.add_op();
  add_op->set_name("add");
  add_op->set_type("QuantizedAdd_8p8to8");
  add_op->set_node_id(1);
  add_op->set_padding(0);
  add_op->add_input("input_node");
  add_op->add_input("add_num");
  add_op->add_input("input_min");
  add_op->add_input("input_max");
  add_op->add_input("add_num_min");
  add_op->add_input("add_num_max");
  add_op->add_output("add:0");
  add_op->add_output("add:1");
  add_op->add_output("add:2");
  NodeInput *input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(0);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(10);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(11);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(12);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(13);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(14);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(15);
  input_node_input->set_output_port(0);
  input_node_input = add_op->add_node_input();
  input_node_input->set_node_id(16);
  input_node_input->set_output_port(0);

  add_op->add_out_max_byte_size(1000);
  add_op->add_out_max_byte_size(1000);
  add_op->add_out_max_byte_size(1000);

  // output op
  OperatorDef *output_op = net.add_op();
  output_op->set_name("__output__");
  output_op->set_type("OUTPUT");
  output_op->set_op_id(2);
  input_node_input = output_op->add_node_input();
  input_node_input->set_node_id(1);
  input_node_input->set_output_port(0);

  // tensor
  TensorProto *add_num_tensor = net.add_tensors();
  add_num_tensor->set_name("add_num");
  add_num_tensor->add_dims(3);
  add_num_tensor->set_data_type(DataType::DT_UINT8);
  add_num_tensor->set_node_id(10);
  add_num_tensor->add_int32_data(0);
  add_num_tensor->add_int32_data(127);
  add_num_tensor->add_int32_data(255);

  TensorProto *input_min_tensor = net.add_tensors();
  input_min_tensor->set_name("input_min");
  input_min_tensor->add_dims(1);
  input_min_tensor->set_data_type(DataType::DT_FLOAT);
  input_min_tensor->set_node_id(11);
  input_min_tensor->add_float_data(-100);

  TensorProto *input_max_tensor = net.add_tensors();
  input_max_tensor->set_name("input_max");
  input_max_tensor->add_dims(1);
  input_max_tensor->set_data_type(DataType::DT_FLOAT);
  input_max_tensor->set_node_id(12);
  input_max_tensor->add_float_data(50.0);

  TensorProto *add_num_min_tensor = net.add_tensors();
  add_num_min_tensor->set_name("add_num_min");
  add_num_min_tensor->add_dims(1);
  add_num_min_tensor->set_data_type(DataType::DT_FLOAT);
  add_num_min_tensor->set_node_id(13);
  add_num_min_tensor->add_float_data(0);

  TensorProto *add_num_max_tensor = net.add_tensors();
  add_num_max_tensor->set_name("add_num_max");
  add_num_max_tensor->add_dims(1);
  add_num_max_tensor->set_data_type(DataType::DT_FLOAT);
  add_num_max_tensor->set_node_id(14);
  add_num_max_tensor->add_float_data(100.0);

  TensorProto *output_min_tensor = net.add_tensors();
  output_min_tensor->set_name("output_min");
  output_min_tensor->add_dims(1);
  output_min_tensor->set_data_type(DataType::DT_FLOAT);
  output_min_tensor->set_node_id(15);
  output_min_tensor->add_float_data(-INFINITY);

  TensorProto *output_max_tensor = net.add_tensors();
  output_max_tensor->set_name("output_max");
  output_max_tensor->add_dims(1);
  output_max_tensor->set_data_type(DataType::DT_FLOAT);
  output_max_tensor->set_node_id(16);
  output_max_tensor->add_float_data(INFINITY);


  // input & output info
  InputInfo *input_info = net.add_input_info();
  input_info->set_name("input_node");
  input_info->set_node_id(0);
  input_info->add_dims(1);
  input_info->add_dims(1);
  input_info->add_dims(1);
  input_info->add_dims(3);
  input_info->set_data_type(DataType::DT_UINT8);
  input_info->set_max_byte_size(1000);
  OutputInfo *output_info = net.add_output_info();
  output_info->set_name("output_node");
  output_info->set_node_id(1);
  output_info->add_dims(1);
  output_info->add_dims(1);
  output_info->add_dims(1);
  output_info->add_dims(3);
  output_info->set_data_type(DataType::DT_UINT8);
  output_info->set_max_byte_size(1000);

  return net;
}

TEST(QuantizedAddTest, QuantizedAdd) {
  testing::internal::LogToStderr();
  HexagonControlWrapper wrapper;
  wrapper.Init();
  wrapper.SetDebugLevel(10);
  wrapper.Config();

  NetDef net = BuildNetDef();
  wrapper.SetupGraph(net);

  Allocator *cpu_allocator = GetDeviceAllocator(DeviceType::CPU);
  Tensor input_tensor(cpu_allocator, DT_UINT8);
  Tensor output_tensor(cpu_allocator, DT_UINT8);
  input_tensor.Resize({1, 1, 1, 3});
  output_tensor.Resize({1, 1, 1, 3});
  uint8_t *input_data = input_tensor.mutable_data<uint8_t>();
  const uint8_t *output_data = output_tensor.data<uint8_t>();
  // [-100.0 0 50] + [0.0, 50.0, 100.0] = [-100.0, 50.0, 150.0]
  // s=0.5859, q0=170, [0, 170, 255]
  // s=0.3906, q0=0, [0, 127, 255]
  input_data[0] = 0;
  input_data[1] = 170;
  input_data[2] = 250;

  VLOG(0) << wrapper.ExecuteGraph(input_tensor, &output_tensor);
  wrapper.PrintLog();

  // -120.0~176.47, [17, 146, 229]
  vector<uint8_t> expected {17, 146, 229};
  for (int i = 0; i < output_tensor.size(); ++i) {
    EXPECT_EQ(expected[i], output_data[i]);
  }

  VLOG(0) << wrapper.TeardownGraph();
  wrapper.Finalize();
}