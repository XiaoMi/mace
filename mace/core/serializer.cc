//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/serializer.h"

namespace mace {

unique_ptr<ConstTensor> Serializer::Serialize(const Tensor &tensor,
                                              const string &name) {
  MACE_NOT_IMPLEMENTED;
  return nullptr;
}

unique_ptr<Tensor> Serializer::Deserialize(const ConstTensor &proto,
                                           DeviceType type) {
  unique_ptr<Tensor> tensor(
      new Tensor(GetDeviceAllocator(type), proto.data_type()));
  vector<index_t> dims;
  for (const index_t d : proto.dims()) {
    dims.push_back(d);
  }
  tensor->Resize(dims);

  switch (proto.data_type()) {
    case DT_FLOAT:
      tensor->Copy<float>(reinterpret_cast<const float *>(proto.data()),
                          proto.data_size());
      break;
    case DT_DOUBLE:
      tensor->Copy<double>(reinterpret_cast<const double *>(proto.data()),
                           proto.data_size());
      break;
    case DT_INT32:
      tensor->Copy<int32_t>(reinterpret_cast<const int32_t *>(proto.data()),
                            proto.data_size());
      break;
    case DT_INT64:
      tensor->Copy<int64_t>(reinterpret_cast<const int64_t *>(proto.data()),
                            proto.data_size());
      break;
    case DT_UINT8:
      tensor->CopyWithCast<int32_t, uint8_t>(
          reinterpret_cast<const int32_t *>(proto.data()), proto.data_size());
      break;
    case DT_INT16:
      tensor->CopyWithCast<int32_t, uint16_t>(
          reinterpret_cast<const int32_t *>(proto.data()), proto.data_size());
      break;
    case DT_INT8:
      tensor->CopyWithCast<int32_t, int8_t>(
          reinterpret_cast<const int32_t *>(proto.data()), proto.data_size());
      break;
    case DT_UINT16:
      tensor->CopyWithCast<int32_t, int16_t>(
          reinterpret_cast<const int32_t *>(proto.data()), proto.data_size());
      break;
    case DT_BOOL:
      tensor->CopyWithCast<int32_t, bool>(
          reinterpret_cast<const int32_t *>(proto.data()), proto.data_size());
      break;
    default:
      MACE_NOT_IMPLEMENTED;
      break;
  }

  return tensor;
}

}  // namespace mace
