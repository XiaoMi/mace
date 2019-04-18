// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <fstream>

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class GatherOpTest : public OpsTestBase {};

namespace {
template<typename T>
void TestGather(const std::vector<index_t> &weight_shape,
                const std::vector<T> &weight,
                const std::vector<index_t> &input_shape,
                const std::vector<int32_t> &input,
                const int axis,
                const std::vector<index_t> &output_shape,
                const std::vector<T> &output) {
  OpsTestNet net;

  net.AddInputFromArray<CPU, T>("Params", weight_shape, weight);
  net.AddInputFromArray<CPU, int32_t>("Indices", input_shape, input);

  OpDefBuilder("Gather", "GatherTest")
      .Input("Params")
      .Input("Indices")
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("axis", axis)
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(CPU);

  auto expected = net.CreateTensor<T>(output_shape, output);

  ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(GatherOpTest, CPUScalarIndex) {
  TestGather<float>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {}, {5}, 0, {2}, {10, 11});
  TestGather<uint8_t>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {}, {5}, 0, {2}, {10, 11});
}

TEST_F(GatherOpTest, CPURank1Index) {
  TestGather<float>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {3}, {2, 4, 6}, 0, {3, 2}, {4, 5, 8, 9, 12, 13});
  TestGather<uint8_t>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {3}, {2, 4, 6}, 0, {3, 2}, {4, 5, 8, 9, 12, 13});
}

TEST_F(GatherOpTest, CPURank1IndexWithAxis1) {
  TestGather<float>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {1}, {1}, 1, {10, 1}, {1, 3, 5, 7, 9, 11, 13, 15, 17, 19});
  TestGather<uint8_t>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {1}, {1}, 1, {10, 1}, {1, 3, 5, 7, 9, 11, 13, 15, 17, 19});
}

TEST_F(GatherOpTest, CPURankHighIndex) {
  TestGather<float>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {1, 3}, {2, 4, 6}, 0, {1, 3, 2}, {4, 5, 8, 9, 12, 13});
  TestGather<uint8_t>({10, 2}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
             {1, 3}, {2, 4, 6}, 0, {1, 3, 2}, {4, 5, 8, 9, 12, 13});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
