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

#ifndef MICRO_BASE_SERIALIZE_H_
#define MICRO_BASE_SERIALIZE_H_

#include <stdint.h>

#include "micro/base/serialize_type.h"
#include "micro/include/public/micro.h"

namespace micro {

#ifdef MACE_WRITE_MAGIC
#ifndef MACE_DEFINE_HARD_CODE_MAGIC
#define MACE_DEFINE_HARD_CODE_MAGIC(CLASS_NAME) \
SerialUint32 GetHardCodeMagic() const {    \
  return Magic(#CLASS_NAME);               \
}
#endif  // MACE_DEFINE_HARD_CODE_MAGIC
#else
#ifndef MACE_DEFINE_HARD_CODE_MAGIC
#define MACE_DEFINE_HARD_CODE_MAGIC(CLASS_NAME)
#endif  // MACE_DEFINE_HARD_CODE_MAGIC
#endif  // MACE_WRITE_MAGIC

// We describe a tensor as an output tensor, but it can also
// be used to represent an input tensor.
struct OpIOInfo {
  uint16_t op_def_idx_;
  uint16_t output_idx_;
};

class Serialize {
#ifdef MACE_WRITE_MAGIC
 public:
  SerialUint32 GetMagic() const;
  MaceStatus MagicToString(SerialUint32 magic, char (&array)[5]) const;

 protected:
  SerialUint32 magic_;

 protected:
  SerialUint32 Magic(const char *bytes4) const;
#endif  // MACE_WRITE_MAGIC

 public:
  void Uint2OpIOInfo(const OpIOInfo *output_info) const;
};

}  // namespace micro

#endif  // MICRO_BASE_SERIALIZE_H_
