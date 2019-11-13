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

#ifndef MICRO_MODEL_ARGUMENT_H_
#define MICRO_MODEL_ARGUMENT_H_

#include "micro/base/serialize.h"

namespace micro {
namespace model {

class Argument : public Serialize {
 public:
  MACE_DEFINE_HARD_CODE_MAGIC(Argument)

  MACE_DECLARE_STRING_FUNC(name);
  MACE_DECLARE_OBJECT_FUNC(float, f);
  MACE_DECLARE_OBJECT_FUNC(int32_t, i);
  MACE_DECLARE_BYTES_FUNC(s);
  MACE_DECLARE_ARRAY_FUNC(float, floats);
  MACE_DECLARE_ARRAY_BASE_PTR_FUNC(float, floats);
  MACE_DECLARE_ARRAY_FUNC(int32_t, ints);
  MACE_DECLARE_ARRAY_BASE_PTR_FUNC(int32_t, ints);

 private:
  SerialString name_;
  SerialFloat f_;
  SerialInt32 i_;
  SerialBytes s_;
  SerialArray<SerialFloat> floats_;
  SerialArray<SerialInt32> ints_;
};

}  // namespace model
}  // namespace micro

#endif  // MICRO_MODEL_ARGUMENT_H_
