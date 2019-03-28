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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DSP_OPS_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DSP_OPS_H_

#include <string>
#include <unordered_map>

#include "mace/utils/logging.h"

namespace mace {

#define MACE_OP_INVALID -1

typedef enum op_type_enum {
#define DEF_OP(NAME, ...) OP_##NAME,

#include "third_party/nnlib/ops.h"  // NOLINT(build/include)
  NN_OPS_MAX

#undef DEF_OP
} op_type;

class OpMap {
 public:
  void Init() {
#define DEF_OP(NAME) op_map_[#NAME] = OP_##NAME;

#include "third_party/nnlib/ops.h"  // NOLINT(build/include)

#undef DEF_OP
  }

  int GetOpId(const std::string &op_type) {
    if (op_map_.find(op_type) != end(op_map_)) {
      return op_map_[op_type];
    } else {
      LOG(ERROR) << "DSP unsupported op type: " << op_type;
      return MACE_OP_INVALID;
    }
  }

 private:
  std::unordered_map<std::string, int> op_map_;
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DSP_OPS_H_
