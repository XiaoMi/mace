//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/hexagon_control_wrapper.h"
#include "gtest/gtest.h"
#include <math.h>

using namespace mace;

static NetDef BuildNetDef() {
  NetDef net;
  net.set_name("supernode_test");
  // input op
  OperatorDef *input_op = net.add_op();
  input_op->set_name("input_node");
  input_op->set_type("INPUT");
  input_op->set_node_id(0);
  input_op->set_padding(0);

  // add op
  OperatorDef *supernode_op = net.add_op();
  supernode_op->set_name("supernode");
  supernode_op->set_type("Supernode_8x8p8to8");
  supernode_op->set_node_id(1);
  supernode_op->set_padding(0);
  supernode_op->add_input("input_node");
  supernode_op->add_input("filter_tensor");
  supernode_op->add_input("input_min");
  supernode_op->add_input("input_max");
  supernode_op->add_input("filter_min");
  supernode_op->add_input("filter_max");
  supernode_op->add_input("stride_tensor");
  supernode_op->add_input("bias_tensor");
  supernode_op->add_input("bias_min");
  supernode_op->add_input("bias_max");
  supernode_op->add_input("min_val");
  supernode_op->add_input("max_val");
  supernode_op->add_output("supernode:0");
  supernode_op->add_output("supernode:1");
  supernode_op->add_output("supernode:2");
  NodeInput *input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(0);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(10);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(11);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(12);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(13);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(14);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(15);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(16);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(17);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(18);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(19);
  input_node_input->set_output_port(0);
  input_node_input = supernode_op->add_node_input();
  input_node_input->set_node_id(20);
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
  TensorProto *filter_tensor = net.add_tensors();
  filter_tensor->set_name("filter_tensor");
  filter_tensor->add_dims(2);
  filter_tensor->add_dims(2);
  filter_tensor->add_dims(1);
  filter_tensor->add_dims(1);
  filter_tensor->set_data_type(DataType::DT_UINT8);
  filter_tensor->set_node_id(10);
  filter_tensor->add_int32_data(0);
  filter_tensor->add_int32_data(127);
  filter_tensor->add_int32_data(127);
  filter_tensor->add_int32_data(255);

  TensorProto *input_min_tensor = net.add_tensors();
  input_min_tensor->set_name("input_min");
  input_min_tensor->add_dims(1);
  input_min_tensor->set_data_type(DataType::DT_FLOAT);
  input_min_tensor->set_node_id(11);
  input_min_tensor->add_float_data(-10.0);

  TensorProto *input_max_tensor = net.add_tensors();
  input_max_tensor->set_name("input_max");
  input_max_tensor->add_dims(1);
  input_max_tensor->set_data_type(DataType::DT_FLOAT);
  input_max_tensor->set_node_id(12);
  input_max_tensor->add_float_data(10.0787402);

  TensorProto *filter_min_tensor = net.add_tensors();
  filter_min_tensor->set_name("filter_min");
  filter_min_tensor->add_dims(1);
  filter_min_tensor->set_data_type(DataType::DT_FLOAT);
  filter_min_tensor->set_node_id(13);
  filter_min_tensor->add_float_data(-10.0);

  TensorProto *filter_max_tensor = net.add_tensors();
  filter_max_tensor->set_name("filter_max");
  filter_max_tensor->add_dims(1);
  filter_max_tensor->set_data_type(DataType::DT_FLOAT);
  filter_max_tensor->set_node_id(14);
  filter_max_tensor->add_float_data(10.0787402);

  TensorProto *stride_tensor = net.add_tensors();
  stride_tensor->set_name("stride");
  stride_tensor->add_dims(1);
  stride_tensor->add_dims(2);
  stride_tensor->add_dims(2);
  stride_tensor->add_dims(1);
  stride_tensor->set_data_type(DataType::DT_INT32);
  stride_tensor->set_node_id(15);

  TensorProto *bias_tensor = net.add_tensors();
  bias_tensor->set_name("bias");
  bias_tensor->add_dims(1);
  bias_tensor->set_data_type(DataType::DT_UINT8);
  bias_tensor->set_node_id(16);
  bias_tensor->add_int32_data(127);

  TensorProto *bias_min_tensor = net.add_tensors();
  bias_min_tensor->set_name("bias_min");
  bias_min_tensor->add_dims(1);
  bias_min_tensor->set_data_type(DataType::DT_FLOAT);
  bias_min_tensor->set_node_id(17);
  bias_min_tensor->add_float_data(-10.0);

  TensorProto *bias_max_tensor = net.add_tensors();
  bias_max_tensor->set_name("bias_max");
  bias_max_tensor->add_dims(1);
  bias_max_tensor->set_data_type(DataType::DT_FLOAT);
  bias_max_tensor->set_node_id(18);
  bias_max_tensor->add_float_data(10.0787402);

  TensorProto *min_val_tensor = net.add_tensors();
  min_val_tensor->set_name("min_val");
  min_val_tensor->add_dims(1);
  min_val_tensor->set_data_type(DataType::DT_FLOAT);
  min_val_tensor->set_node_id(19);
  min_val_tensor->add_float_data(-INFINITY);

  TensorProto *max_val_tensor = net.add_tensors();
  max_val_tensor->set_name("max_val");
  max_val_tensor->add_dims(1);
  max_val_tensor->set_data_type(DataType::DT_FLOAT);
  max_val_tensor->set_node_id(20);
  max_val_tensor->add_float_data(INFINITY);

  // input & output info
  InputInfo *input_info = net.add_input_info();
  input_info->set_name("input_node");
  input_info->set_node_id(0);
  input_info->add_dims(1);
  input_info->add_dims(4);
  input_info->add_dims(4);
  input_info->add_dims(1);
  input_info->set_data_type(DataType::DT_UINT8);
  input_info->set_max_byte_size(1000);
  OutputInfo *output_info = net.add_output_info();
  output_info->set_name("output_node");
  output_info->set_node_id(1);
  output_info->add_dims(1);
  output_info->add_dims(2);
  output_info->add_dims(2);
  output_info->add_dims(1);
  output_info->set_data_type(DataType::DT_UINT8);
  output_info->set_max_byte_size(1000);

  return net;
}

TEST(SupernodeTest, Supernode) {
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
  input_tensor.Resize({1, 4, 4, 1});
  output_tensor.Resize({1, 2, 2, 1});
  uint8_t *input_data = input_tensor.mutable_data<uint8_t>();
  const uint8_t *output_data = output_tensor.data<uint8_t>();
  // input: [[-10, ..], [-5.03937, ..], [0, ..], [5.03937, ..]]
  // filt: [[-10, 0], [0, 10.07874]]
  // bias: 0.0
  for (int h = 0; h < 4; ++h) {
    for (int w = 0; w < 4; ++w)
      input_data[h * 4 + w] = (uint8_t)((h == 0) ? 0 : h * 64 - 1);
  }

  VLOG(0) << wrapper.ExecuteGraph(input_tensor, &output_tensor);
  wrapper.PrintLog();

  // expect out: [[49.2095, 49.2095], [50.7905, 50.7905]]
  // with output range(-0.5, 64.0), quantize to [[196, 196], [203, 203]]
  vector<uint8_t> expected {196, 196, 203, 203};
  for (int i = 0; i < output_tensor.size(); ++i) {
    EXPECT_EQ(expected[i], output_data[i]);
  }
  std::cout << std::endl;

  VLOG(0) << wrapper.TeardownGraph();
  wrapper.Finalize();
}
