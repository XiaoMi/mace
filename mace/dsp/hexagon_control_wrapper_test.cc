//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/hexagon_control_wrapper.h"
#include "mace/core/logging.h"
#include "gtest/gtest.h"

using namespace mace;

TEST(HexagonControlerWrapper, GetVersion) {
  testing::internal::LogToStderr();
  HexagonControlWrapper wrapper;
  VLOG(0) << "version: " << wrapper.GetVersion();
  wrapper.Init();
  wrapper.SetDebugLevel(3);
  wrapper.Config();
  VLOG(0) << wrapper.SetupGraph("quantized_test_dsp.pb");
  wrapper.PrintGraph();

  Tensor input_tensor;
  Tensor output_tensor;
  input_tensor.Resize({1, 28, 28, 3});
  float *input_data = input_tensor.mutable_data<float>();
  for (int i = 0; i < input_tensor.size(); ++i) {
    input_data[i] = i;
  }

  VLOG(0) << wrapper.ExecuteGraph(input_tensor, &output_tensor);
  wrapper.PrintLog();
  wrapper.GetPerfInfo();

  const float *output_data = output_tensor.data<float>();
  VLOG(0) << output_tensor.size() << output_tensor.dtype();
  for (int i = 0; i < output_tensor.size(); ++i) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  VLOG(0) << wrapper.TeardownGraph();
  wrapper.Finalize();
}