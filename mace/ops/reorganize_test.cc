//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gmock/gmock.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ReOrganizeTest : public OpsTestBase {};

void TestReOrganize(const std::vector<index_t> &input_shape,
                    const std::vector<float> &input_data,
                    const std::vector<index_t> &output_shape,
                    const std::vector<float> &output_data) {
  const std::vector<int> out_shape(output_shape.begin(), output_shape.end());

  // Construct graph
  OpsTestNet net;

  OpDefBuilder("ReOrganize", "ReOrganizeTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("shape", out_shape)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>("Input",
                                                input_shape, input_data);

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");

  EXPECT_THAT(output->shape(), ::testing::ContainerEq(output_shape));

  const float *output_ptr = output->data<float>();
  int size = output->size();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output_data[i], output_ptr[i]) << "With Index " << i;
  }

  // Reverse reorganzie
  const std::vector<int> in_shape(input_shape.begin(), input_shape.end());
  OpDefBuilder("ReOrganize", "ReOrganizeTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("shape", in_shape)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>("Input",
                                                output_shape, output_data);

  // Run
  net.RunOp();

  output = net.GetTensor("Output");

  EXPECT_THAT(output->shape(), ::testing::ContainerEq(input_shape));

  output_ptr = output->data<float>();
  size = output->size();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input_data[i], output_ptr[i]) << "With Index " << i;
  }
}

TEST_F(ReOrganizeTest, Simple) {
  TestReOrganize({1, 1, 4, 6},
                 {0, 4, 8, 12, 16, 20,
                  1, 5, 9, 13, 17, 21,
                  2, 6, 10, 14, 18, 22,
                  3, 7, 11, 15, 19, 23},
                 {1, 1, 8, 3},
                 {0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19,
                  4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23});
  TestReOrganize({1, 1, 5, 6},
                 {0, 5, 10, 15, 20, 25,
                  1, 6, 11, 16, 21, 26,
                  2, 7, 12, 17, 22, 27,
                  3, 8, 13, 18, 23, 28,
                  4, 9, 14, 19, 24, 29},
                 {1, 1, 10, 3},
                 {0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23,
                  4, 14, 24, 5, 15, 25, 6, 16, 26, 7, 17, 27,
                  8, 18, 28, 9, 19, 29});
}

TEST_F(ReOrganizeTest, Complex) {
  TestReOrganize({1, 2, 2, 6},
                 {0, 4, 8, 12, 16, 20,
                  1, 5, 9, 13, 17, 21,
                  2, 6, 10, 14, 18, 22,
                  3, 7, 11, 15, 19, 23},
                 {1, 2, 6, 2},
                 {0, 12, 1, 13, 4, 16, 5, 17, 8, 20, 9, 21,
                  2, 14, 3, 15, 6, 18, 7, 19, 10, 22, 11, 23});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
