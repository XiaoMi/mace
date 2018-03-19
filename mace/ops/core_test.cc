//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

TEST(CoreTest, INIT_MODE) {
  std::vector<OperatorDef> op_defs;

  Workspace ws;

  op_defs.emplace_back(OperatorDef());
  OpDefBuilder("BufferToImage", "BufferToImageTest")
      .Input("Input")
      .Output("B2IOutput")
      .AddIntArg("buffer_type", kernels::BufferType::CONV2D_FILTER)
      .AddIntArg("mode", static_cast<int>(NetMode::INIT))
      .Finalize(&op_defs[op_defs.size() - 1]);

  Tensor *input =
      ws.CreateTensor("Input", GetDeviceAllocator(DeviceType::OPENCL),
                      DataTypeToEnum<float>::v());
  input->Resize({1, 3, 3, 3});
  {
    Tensor::MappingGuard input_mapper(input);
    float *input_data = input->mutable_data<float>();
    std::fill(input_data, input_data + input->size(), 1);
  }

  op_defs.emplace_back(OperatorDef());
  OpDefBuilder("ImageToBuffer", "ImageToBufferTest")
      .Input("B2IOutput")
      .Output("Output")
      .AddIntArg("buffer_type", kernels::BufferType::CONV2D_FILTER)
      .Finalize(&op_defs[op_defs.size() - 1]);

  NetDef net_def;
  for (auto &op_def : op_defs) {
    net_def.add_op()->CopyFrom(op_def);
  }
  std::shared_ptr<OperatorRegistry> op_registry(new OperatorRegistry());
  auto net =
      CreateNet(op_registry, net_def, &ws, DeviceType::OPENCL, NetMode::INIT);
  net->Run();

  EXPECT_TRUE(ws.GetTensor("B2IOutput") != nullptr);
  EXPECT_TRUE(ws.GetTensor("Output") == nullptr);

  net = CreateNet(op_registry, net_def, &ws, DeviceType::OPENCL);
  net->Run();
  EXPECT_TRUE(ws.GetTensor("Output") != nullptr);

  ExpectTensorNear<float>(*ws.GetTensor("Input"), *ws.GetTensor("Output"),
                          1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
