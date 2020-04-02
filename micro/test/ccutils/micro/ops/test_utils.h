// Copyright 2020 The MACE Authors. All Rights Reserved.
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


#ifndef MICRO_TEST_CCUTILS_MICRO_OPS_TEST_UTILS_H_
#define MICRO_TEST_CCUTILS_MICRO_OPS_TEST_UTILS_H_

#include "micro/base/logging.h"
#include "micro/common/global_buffer.h"
#include "micro/include/public/micro.h"
#include "micro/port/api.h"

namespace micro {
namespace ops {
namespace test {

void PrintDims(const int32_t *dims, const uint32_t dim_size);

void AssertSameDims(const int32_t *x_dims, const uint32_t x_dim_size,
                    const int32_t *y_dims, const uint32_t y_dim_size);

void FillRandomInput(void *input, const int32_t shape_size);

#ifndef MACE_DEFINE_RANDOM_INPUT
#define MACE_DEFINE_RANDOM_INPUT(T, input, shape_size)                \
T *input = common::test::GetGlobalBuffer()->GetBuffer<T>(shape_size); \
micro::ops::test::FillRandomInput(input, shape_size * sizeof(T))
#endif

}  // namespace test
}  // namespace ops
}  // namespace micro

#endif  // MICRO_TEST_CCUTILS_MICRO_OPS_TEST_UTILS_H_

