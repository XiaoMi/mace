//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"

#include "mace/core/operator.h"
#include "mace/core/net.h"

using namespace mace;

TEST(PoolingTest, Pooling) {
  OperatorRegistry* registry = gDeviceTypeRegistry()->at(DeviceType::CPU);
  vector<string> registry_keys = registry->Keys();
  for (auto& key: registry_keys) {
    VLOG(0) << "registry_op: " << key;
  }

  // Construct graph
  OperatorDef op_def;
  op_def.add_input("Input0");
  op_def.add_output("Output0");
  op_def.set_name("PoolingTest");
  op_def.set_type("Pooling");
  auto pooling_type = op_def.add_arg();
  pooling_type->set_name("pooling_type");
  pooling_type->set_i(0);
  auto kernel_size = op_def.add_arg();
  kernel_size->set_name("kernel_size");
  kernel_size->set_i(2);
  auto stride = op_def.add_arg();
  stride->set_name("stride");
  stride->set_i(2);
  auto padding = op_def.add_arg();
  padding->set_name("padding");
  padding->set_i(0);

  NetDef net_def;
  net_def.set_name("NetTest");
  net_def.add_op()->CopyFrom(op_def);

  VLOG(0) << net_def.DebugString();

  // Create workspace and input tensor
  Workspace ws;
  Tensor* input = ws.CreateTensor("Input0", cpu_allocator(), DataType::DT_FLOAT);
  Tensor* output = ws.CreateTensor("Output0", cpu_allocator(), DataType::DT_FLOAT);
  input->Resize({2, 2, 4, 4});
  float* input_data = input->mutable_data<float>();
  for (int i = 0; i < 64; ++i) {
    input_data[i] = i;
  }

  // Create Net & run
  auto net = CreateNet(net_def, &ws, DeviceType::CPU);
  net->Run();

  for (int d :output->shape()){
    ASSERT_EQ(d, 2);
  }

  ASSERT_FLOAT_EQ(output->data<float>()[0], 5);
  ASSERT_FLOAT_EQ(output->data<float>()[3], 15);
  ASSERT_FLOAT_EQ(output->data<float>()[15], 63);
}