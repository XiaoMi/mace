//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_HEXAGON_NN_OPS_H_
#define MACE_HEXAGON_NN_OPS_H_

#include "mace/core/logging.h"
#include <unordered_map>

namespace mace {

#define OP_INVALID -1

typedef enum op_type_enum {
#define DEF_OP(NAME, ...) OP_##NAME,

#include "mace/dsp/ops.h"
  NN_OPS_MAX

#undef DEF_OP
} op_type;

class OpMap {
 public:
  void Init() {
#define DEF_OP(NAME) \
    op_map_[#NAME] = OP_##NAME;

#include "mace/dsp/ops.h"

#undef DEF_OP
  }

  int GetOpId(std::string op_type) {
    if (op_map_.find(op_type) != end(op_map_)) {
      return op_map_[op_type];
    } else {
      LOG(ERROR) << "DSP unsupoorted op type: " << op_type;
      return OP_INVALID;
    }
  }
 private:
  std::unordered_map<std::string, int> op_map_;
};
} // namespace mace

#endif // MACE_HEXAGON_NN_OPS_H_
