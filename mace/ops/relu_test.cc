//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"

#include "mace/core/operator.h"
#include "mace/core/net.h"

using namespace mace;

TEST(ReluTest, Relu) {
  OperatorRegistry* registry = gDeviceTypeRegistry()->at(DeviceType::CPU);
  vector<string> registry_keys = registry->Keys();
  for (auto& key: registry_keys) {
    VLOG(0) << "registry_op: " << key;
  }

  // Construct graph
  OperatorDef op_def;
  op_def.add_input("Input0");
  op_def.add_output("Output0");
  op_def.set_name("ReluTest");
  op_def.set_type("Relu");
  auto arg = op_def.add_arg();
  arg->set_name("arg0");
  arg->set_f(1.5);

  NetDef net_def;
  net_def.set_name("NetTest");
  net_def.add_op()->CopyFrom(op_def);

  VLOG(0) << net_def.DebugString();

  // Create workspace and input tensor
  Workspace ws;
  Tensor* input = ws.CreateTensor("Input0", cpu_allocator(), DataType::DT_FLOAT);
  input->Resize({2,3});
  float* input_data = input->mutable_data<float>();
  for (int i = 0; i < 6; ++i) {
    input_data[i] = i-3;
  }

  // Create Net & run
  auto net = CreateNet(net_def, &ws, DeviceType::CPU);
  net->Run();

  // Create Op & run
  auto op = CreateOperator(op_def, &ws, DeviceType::CPU);
  ASSERT_FLOAT_EQ(1.5f, op->GetSingleArgument<float>("arg0", 1.0f));

}