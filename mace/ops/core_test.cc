// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

TEST(CoreTest, INIT_MODE) {
  std::vector<OperatorDef> op_defs;

  Device *device = OpTestContext::Get()->GetDevice(DeviceType::GPU);
  std::unique_ptr<Tuner<uint32_t>> tuner;
  Workspace ws;

  op_defs.emplace_back(OperatorDef());
  OpDefBuilder("BufferTransform", "BufferTransformTest")
      .Input("Input")
      .Output("B2IOutput")
      .AddIntArg("buffer_type", kernels::BufferType::CONV2D_FILTER)
      .AddIntArg("mode", static_cast<int>(NetMode::INIT))
      .Finalize(&op_defs[op_defs.size() - 1]);

  Tensor *input = ws.CreateTensor("Input", device->allocator(),
                                  DataTypeToEnum<float>::v());
  input->Resize({1, 3, 3, 3});
  {
    Tensor::MappingGuard input_mapper(input);
    float *input_data = input->mutable_data<float>();
    std::fill(input_data, input_data + input->size(), 1);
  }

  op_defs.emplace_back(OperatorDef());
  OpDefBuilder("BufferInverseTransform", "BufferInverseTransformTest")
      .Input("B2IOutput")
      .Output("Output")
      .AddIntArg("buffer_type", kernels::BufferType::CONV2D_FILTER)
      .Finalize(&op_defs[op_defs.size() - 1]);

  NetDef net_def;
  for (auto &op_def : op_defs) {
    net_def.add_op()->CopyFrom(op_def);
    net_def.add_op_types(op_def.type());
  }
  std::shared_ptr<OpDefRegistryBase> op_def_registry(new OpDefRegistry());
  std::shared_ptr<OpRegistryBase> op_registry(new OpRegistry());
  auto net = std::unique_ptr<NetBase>(new SerialNet(
      op_def_registry.get(), op_registry.get(), &net_def, &ws, device,
      NetMode::INIT));
  MaceStatus status = net->Init();
  MACE_CHECK(status == MaceStatus::MACE_SUCCESS);
  status = net->Run();
  MACE_CHECK(status == MaceStatus::MACE_SUCCESS);

  EXPECT_TRUE(ws.GetTensor("B2IOutput") != nullptr);
  EXPECT_TRUE(ws.GetTensor("Output") == nullptr);
  net = std::unique_ptr<NetBase>(new SerialNet(
      op_def_registry.get(), op_registry.get(), &net_def, &ws, device));
  status = net->Init();
  MACE_CHECK(status == MaceStatus::MACE_SUCCESS);
  status = net->Run();
  MACE_CHECK(status == MaceStatus::MACE_SUCCESS);
  EXPECT_TRUE(ws.GetTensor("Output") != nullptr);

  ExpectTensorNear<float>(*ws.GetTensor("Input"), *ws.GetTensor("Output"),
                          1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
