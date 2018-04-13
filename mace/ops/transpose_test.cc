//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class TransposeOpTest : public OpsTestBase {};

namespace {
void TransposeNCHWTest(const std::vector<index_t> &input_shape) {
  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<CPU, float>("Input", input_shape);

  OpDefBuilder("Transpose", "TransposeNCHWTest")
    .Input("Input")
    .Output("Output")
    .AddIntsArg("dims", {0, 3, 1, 2})
    .Finalize(net.NewOperatorDef());

  // Run on cpu
  net.RunOp();

  net.FillNHWCInputToNCHWInput<DeviceType::CPU, float>("InputNCHW", "Input");

  ExpectTensorNear<float>(*net.GetOutput("InputNCHW"),
                          *net.GetOutput("Output"),
                          0.01);
}
}  // namespace

TEST_F(TransposeOpTest, NCHW) {
  TransposeNCHWTest({3, 64, 64, 128});
  TransposeNCHWTest({1, 64, 48, 128});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
