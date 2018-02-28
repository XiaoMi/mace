//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/serializer.h"

namespace mace {

std::unique_ptr<ConstTensor> Serializer::Serialize(const Tensor &tensor,
                                                   const std::string &name) {
  MACE_NOT_IMPLEMENTED;
  return nullptr;
}

std::unique_ptr<Tensor> Serializer::Deserialize(const ConstTensor &const_tensor,
                                                DeviceType type) {
  std::unique_ptr<Tensor> tensor(
      new Tensor(GetDeviceAllocator(type), const_tensor.data_type()));
  std::vector<index_t> dims;
  for (const index_t d : const_tensor.dims()) {
    dims.push_back(d);
  }
  tensor->Resize(dims);

  switch (const_tensor.data_type()) {
    case DT_HALF:
      tensor->Copy<half>(reinterpret_cast<const half *>(const_tensor.data()),
                         const_tensor.data_size());
      break;
    case DT_FLOAT:
      tensor->Copy<float>(reinterpret_cast<const float *>(const_tensor.data()),
                          const_tensor.data_size());
      break;
    case DT_DOUBLE:
      tensor->Copy<double>(
          reinterpret_cast<const double *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_INT32:
      tensor->Copy<int32_t>(
          reinterpret_cast<const int32_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_INT64:
      tensor->Copy<int64_t>(
          reinterpret_cast<const int64_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_UINT8:
      tensor->Copy<uint8_t>(
          reinterpret_cast<const uint8_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_INT16:
      tensor->CopyWithCast<int32_t, uint16_t>(
          reinterpret_cast<const int32_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_INT8:
      tensor->CopyWithCast<int32_t, int8_t>(
          reinterpret_cast<const int32_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_UINT16:
      tensor->CopyWithCast<int32_t, int16_t>(
          reinterpret_cast<const int32_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    case DT_BOOL:
      tensor->CopyWithCast<int32_t, bool>(
          reinterpret_cast<const int32_t *>(const_tensor.data()),
          const_tensor.data_size());
      break;
    default:
      MACE_NOT_IMPLEMENTED;
      break;
  }

  return tensor;
}

}  // namespace mace
