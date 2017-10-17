//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_SERIALIZER_H_
#define MACE_CORE_SERIALIZER_H_

#include "mace/core/common.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class Serializer {
 public:
  Serializer() {}
  ~Serializer() {}

  unique_ptr<TensorProto> Serialize(const Tensor &tensor, const string &name);

  unique_ptr<Tensor> Deserialize(const TensorProto &proto, DeviceType type);

  DISABLE_COPY_AND_ASSIGN(Serializer);
};

}  // namespace mace

#endif  // MACE_CORE_SERIALIZER_H_
