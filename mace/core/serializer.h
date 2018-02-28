//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_SERIALIZER_H_
#define MACE_CORE_SERIALIZER_H_

#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {

class Serializer {
 public:
  Serializer() {}
  ~Serializer() {}

  std::unique_ptr<ConstTensor> Serialize(const Tensor &tensor,
                                         const std::string &name);

  std::unique_ptr<Tensor> Deserialize(const ConstTensor &const_tensor,
                                      DeviceType type);

  DISABLE_COPY_AND_ASSIGN(Serializer);
};

}  // namespace mace

#endif  // MACE_CORE_SERIALIZER_H_
