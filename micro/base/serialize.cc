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

#include "micro/base/serialize.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"

namespace micro {

#ifdef MACE_WRITE_MAGIC
SerialUint32 Serialize::GetMagic() const {
  return magic_;
}

SerialUint32 Serialize::Magic(const char *bytes4) const {
  MACE_ASSERT1(micro::base::strlen(bytes4) >= 4, "The magic bytes must >= 4.");
  SerialUint32 magic = 0;
  for (int32_t i = 0; i < 32 && (*bytes4) != '\0'; i += 8, ++bytes4) {
    magic += (*bytes4) << i;
  }
  return magic;
}

MaceStatus Serialize::MagicToString(SerialUint32 magic,
                                    char (&array)[5]) const {
  char *buffer = array;
  for (int32_t i = 0; i <32; i += 8, ++buffer) {
    *buffer = (magic >> i) & 0x000000ff;
  }
  *buffer = '\0';
  return MACE_SUCCESS;
}
#endif  // MACE_WRITE_MAGIC

void Serialize::Uint2OpIOInfo(const OpIOInfo *info) const {
  OpIOInfo *io_info = const_cast<OpIOInfo *>(info);
  uint32_t info_data = *(reinterpret_cast<uint32_t *>(io_info));
  io_info->op_def_idx_ = (info_data & 0xffff0000) >> 16;
  io_info->output_idx_ = (info_data & 0x0000ffff);
}

}  // namespace micro
