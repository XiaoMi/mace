//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/net.h"

using namespace mace;

int main() {
  // Construct graph
  OperatorDef op_def_0;
  op_def_0.add_input("Input");
  op_def_0.add_output("Output0");
  op_def_0.set_name("ReluTest0");
  op_def_0.set_type("Relu");
  auto arg_0 = op_def_0.add_arg();
  arg_0->set_name("arg0");
  arg_0->set_f(0.5);

  OperatorDef op_def_1;
  op_def_1.add_input("Input");
  op_def_1.add_output("Output1");
  op_def_1.set_name("ReluTest1");
  op_def_1.set_type("Relu");
  auto arg_1 = op_def_1.add_arg();
  arg_1->set_name("arg0");
  arg_1->set_f(1.5);

  OperatorDef op_def_2;
  op_def_2.add_input("Output1");
  op_def_2.add_output("Output2");
  op_def_2.set_name("ReluTest2");
  op_def_2.set_type("Relu");
  auto arg_2 = op_def_2.add_arg();
  arg_2->set_name("arg0");
  arg_2->set_f(2.5);

  NetDef net_def;
  net_def.set_name("NetTest");
  net_def.add_op()->CopyFrom(op_def_0);
  net_def.add_op()->CopyFrom(op_def_1);
  net_def.add_op()->CopyFrom(op_def_2);

  auto input = net_def.add_tensors();
  input->set_name("Input");
  input->set_data_type(DataType::DT_FLOAT);
  input->add_dims(2);
  input->add_dims(3);
  for (int i = 0; i < 6; ++i) {
    input->add_float_data(i - 3);
  }

  VLOG(0) << net_def.DebugString();

  // Create workspace and input tensor
  Workspace ws;
  ws.LoadModelTensor(net_def, DeviceType::CPU);

  // Create Net & run
  auto net = CreateNet(net_def, &ws, DeviceType::CPU);
  net->Run();

  auto out_tensor = ws.GetTensor("Output2");
  out_tensor->DebugPrint();

  return 0;
}
