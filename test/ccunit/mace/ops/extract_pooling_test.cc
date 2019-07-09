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

#include "gmock/gmock.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ExtractPoolingTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestExtractPooling(const std::vector<index_t> &input_shape,
                        const std::vector<float> &input_value,
                        const int modulus,
                        const int num_log_count,
                        const int include_variance,
                        const std::vector<int> &forward_indexes,
                        const std::vector<float> &counts,
                        const std::vector<index_t> &output_shape,
                        const std::vector<float> &output_value) {
  // Construct graph
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input", input_shape, input_value);
  OpDefBuilder("ExtractPooling", "ExtractPoolingTest")
      .Input("Input")
      .AddIntArg("modulus", modulus)
      .AddIntArg("include_variance", include_variance)
      .AddIntArg("num_log_count", num_log_count)
      .AddIntsArg("forward_indexes", forward_indexes)
      .AddFloatsArg("counts", counts)
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp();
  // Check
  auto expected = net.CreateTensor<float>(output_shape, output_value);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ExtractPoolingTest, SimpleCPU) {
  TestExtractPooling<DeviceType::CPU, float>(
    {3, 20, 3},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
     61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
     76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
     91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
     106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
     121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
     136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
     151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
     166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179},
    9, 0, 0,
    {0, 6, 2, 6},
    {6, 4},
    {3, 2, 3},
    {7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
     67.5, 68.5, 69.5, 70.5, 71.5, 72.5,
     127.5, 128.5, 129.5, 130.5, 131.5, 132.5});
}

TEST_F(ExtractPoolingTest, SimpleCPUWithVariance) {
TestExtractPooling<DeviceType::CPU, float>(
    {3, 20, 3},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
     61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
     76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
     91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
     106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
     121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
     136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
     151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
     166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179},
     9, 1, 1,
    {0, 6, 2, 6},
    {6, 4},
    {3, 2, 7},
    {1.7917595, 7.5, 8.5, 9.5, 5.1234756, 5.1234756, 5.1234756,
     1.3862944, 10.5, 11.5, 12.5, 3.354102, 3.354102, 3.354102,
     1.7917595, 67.5, 68.5, 69.5, 5.1234756, 5.1234756, 5.1234756,
     1.3862944, 70.5, 71.5, 72.5, 3.354102, 3.354102, 3.354102,
     1.7917595, 127.5, 128.5, 129.5, 5.1234756, 5.1234756, 5.1234756,
     1.3862944, 130.5, 131.5, 132.5, 3.354102, 3.354102, 3.354102});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
