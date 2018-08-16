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

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class StridedSliceOpTest : public OpsTestBase {};

namespace {

void TestStridedSlice(const std::vector<index_t> &input_shape,
                      const std::vector<float> &input,
                      const std::vector<int32_t> &begin_indices,
                      const std::vector<int32_t> &end_indices,
                      const std::vector<int32_t> &strides,
                      const int begin_mask,
                      const int end_mask,
                      const int ellipsis_mask,
                      const int new_axis_mask,
                      const int shrink_axis_mask,
                      const std::vector<index_t> &output_shape,
                      const std::vector<float> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input", input_shape, input);
  net.AddInputFromArray<CPU, int32_t>(
      "BeginIndices", {static_cast<int32_t>(begin_indices.size())},
      begin_indices);
  net.AddInputFromArray<CPU, int32_t>(
      "EndIndices", {static_cast<int32_t>(end_indices.size())}, end_indices);
  net.AddInputFromArray<CPU, int32_t>(
      "Strides", {static_cast<int32_t>(strides.size())}, strides);

  OpDefBuilder("StridedSlice", "StridedSliceOpTest")
      .Input("Input")
      .Input("BeginIndices")
      .Input("EndIndices")
      .Input("Strides")
      .Output("Output")
      .AddIntArg("begin_mask", begin_mask)
      .AddIntArg("end_mask", end_mask)
      .AddIntArg("ellipsis_mask", ellipsis_mask)
      .AddIntArg("new_axis_mask", new_axis_mask)
      .AddIntArg("shrink_axis_mask", shrink_axis_mask)
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, float>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                          *net.GetOutput("Output"));
}

void TestSlice(const std::vector<index_t> &input_shape,
               const std::vector<float> &input,
               const std::vector<int32_t> &begin_indices,
               const std::vector<int32_t> &indices_size,
               const std::vector<index_t> &output_shape,
               const std::vector<float> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, float>("Input", input_shape, input);
  net.AddInputFromArray<CPU, int32_t>(
      "BeginIndices", {static_cast<int32_t>(input_shape.size())},
      begin_indices);
  net.AddInputFromArray<CPU, int32_t>(
      "IndicesSize", {static_cast<int32_t>(indices_size.size())}, indices_size);

  OpDefBuilder("StridedSlice", "StridedSliceOpTest")
      .Input("Input")
      .Input("BeginIndices")
      .Input("IndicesSize")
      .Output("Output")
      .AddIntArg("slice", 1)
      .Finalize(net.NewOperatorDef());

  net.RunOp();
  net.AddInputFromArray<CPU, float>("ExpectedOutput", output_shape, output);
  ExpectTensorNear<float>(*net.GetOutput("ExpectedOutput"),
                          *net.GetOutput("Output"));
}

}  // namespace

TEST_F(StridedSliceOpTest, TestStridedSliceByFirstAxis) {
  TestStridedSlice({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                   {1, 0, 0}, {2, 3, 2}, {1, 1, 1}, 0, 0, 0, 0, 0, {1, 3, 2},
                   {7, 8, 9, 10, 11, 12});
  TestStridedSlice({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                   {1, 0, 0}, {2, 3, 2}, {1, 1, 1}, 0, 0, 0, 0, 1, {3, 2},
                   {7, 8, 9, 10, 11, 12});
  TestStridedSlice({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                   {1, 1, 2}, {2, 3, 2}, {1, 1, 1}, 6, 6, 0, 0, 0, {1, 3, 2},
                   {7, 8, 9, 10, 11, 12});
}

TEST_F(StridedSliceOpTest, TestStridedSliceRank1) {
  TestStridedSlice({4}, {1, 2, 3, 4}, {1}, {3}, {1}, 0, 0, 0, 0, 0, {2},
                   {2, 3});
  TestStridedSlice({4}, {1, 2, 3, 4}, {-3}, {3}, {1}, 0, 0, 0, 0, 0, {2},
                   {2, 3});
  TestStridedSlice({4}, {1, 2, 3, 4}, {-2}, {-4}, {-1}, 0, 0, 0, 0, 0, {2},
                   {3, 2});
  TestStridedSlice({4}, {1, 2, 3, 4}, {-1}, {-4}, {-2}, 0, 0, 0, 0, 0, {2},
                   {4, 2});
  TestStridedSlice({4}, {1, 2, 3, 4}, {-2}, {-4}, {-1}, 1, 0, 0, 0, 0, {3},
                   {4, 3, 2});
  TestStridedSlice({4}, {1, 2, 3, 4}, {-2}, {-4}, {-1}, 0, 1, 0, 0, 0, {3},
                   {3, 2, 1});
  TestStridedSlice({4}, {1, 2, 3, 4}, {-2}, {-4}, {-1}, 1, 1, 0, 0, 0, {4},
                   {4, 3, 2, 1});
  TestStridedSlice({4}, {1, 2, 3, 4}, {2}, {4}, {2}, 1, 1, 0, 0, 0, {2},
                   {1, 3});
  TestStridedSlice({4}, {1, 2, 3, 4}, {2}, {3}, {1}, 0, 0, 0, 0, 1, {}, {3});
}

TEST_F(StridedSliceOpTest, TestStridedSliceRank2) {
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {0, 0}, {2, 3}, {1, 1}, 0, 0, 0,
                   0, 0, {2, 3}, {1, 2, 3, 4, 5, 6});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {0}, {2}, {1}, 0, 0, 0,
                   0, 0, {2, 3}, {1, 2, 3, 4, 5, 6});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1}, {2, 3}, {1, 1}, 0, 0, 0,
                   0, 0, {1, 2}, {5, 6});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {0, 0}, {2, 3}, {1, 2}, 0, 0, 0,
                   0, 0, {2, 2}, {1, 3, 4, 6});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2}, {0, 0}, {-1, -1}, 0, 0,
                   0, 0, 0, {1, 2}, {6, 5});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2}, {0, 0}, {-1, -1}, 3, 3,
                   0, 0, 0, {2, 3}, {6, 5, 4, 3, 2, 1});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 0}, {2, 3}, {1, 1}, 0, 0, 0,
                   0, 1, {3}, {4, 5, 6});
  TestStridedSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2}, {2, 3}, {1, 1}, 0, 0, 0,
                   0, 3, {}, {6});
}

TEST_F(StridedSliceOpTest, TestStridedSliceRank3) {
  TestStridedSlice({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                   {0, 0, 0}, {2, 3, 2}, {1, 2, 1}, 0, 0, 0, 0, 0, {2, 2, 2},
                   {1, 2, 5, 6, 7, 8, 11, 12});
  TestStridedSlice({3, 2, 3}, {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6,
                   6, 6}, {1, 0, 0}, {2, 1, 3}, {1, 1, 1}, 0, 0, 0, 0, 0, {1,
                   1, 3}, {3, 3, 3});
  TestStridedSlice({3, 2, 3}, {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6,
                   6, 6}, {0, 0, 0}, {2, 2, 2}, {1, 2, 1}, 0, 0, 0, 0, 0, {2,
                   1, 2}, {1, 1, 3, 3});
}

TEST_F(StridedSliceOpTest, TestSlice) {
  TestSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {0, 0}, {2, 3}, {2, 3},
            {1, 2, 3, 4, 5, 6});
  TestSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 0}, {1, 2}, {1, 2}, {4, 5});
  TestSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {0, 0}, {2, -1}, {2, 3},
            {1, 2, 3, 4, 5, 6});
  TestSlice({2, 3}, {1, 2, 3, 4, 5, 6}, {0, 1}, {2, -1}, {2, 2}, {2, 3, 5, 6});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
