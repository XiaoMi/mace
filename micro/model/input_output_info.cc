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

#include "micro/model/input_output_info.h"

namespace micro {
namespace model {

MACE_DEFINE_STRING_FUNC(InputOutputInfo, name, name_)
MACE_DEFINE_OBJECT_FUNC(InputOutputInfo, int32_t, node_id)
MACE_DEFINE_ARRAY_FUNC(InputOutputInfo, int32_t, dim, dims_)
MACE_DEFINE_OBJECT_FUNC(InputOutputInfo, int32_t, max_byte_size)
MACE_DEFINE_OBJECT_FUNC(InputOutputInfo, int32_t, data_type)
MACE_DEFINE_OBJECT_FUNC(InputOutputInfo, int32_t, data_format)
MACE_DEFINE_OBJECT_FUNC(InputOutputInfo, float, scale)
MACE_DEFINE_OBJECT_FUNC(InputOutputInfo, int32_t, zero_point)

}  // namespace model
}  // namespace micro
