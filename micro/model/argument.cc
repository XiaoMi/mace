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

#include "micro/model/argument.h"

namespace micro {
namespace model {

MACE_DEFINE_STRING_FUNC(Argument, name, name_)
MACE_DEFINE_OBJECT_FUNC(Argument, float, f)
MACE_DEFINE_OBJECT_FUNC(Argument, int32_t, i)
MACE_DEFINE_BYTES_FUNC(Argument, s, s_)
MACE_DEFINE_ARRAY_FUNC(Argument, float, floats, floats_)
MACE_DEFINE_ARRAY_BASE_PTR_FUNC(Argument, float, floats, floats_)
MACE_DEFINE_ARRAY_FUNC(Argument, int32_t, ints, ints_)
MACE_DEFINE_ARRAY_BASE_PTR_FUNC(Argument, int32_t, ints, ints_)

}  // namespace model
}  // namespace micro
