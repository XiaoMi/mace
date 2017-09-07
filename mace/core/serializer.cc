//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/serializer.h"


namespace mace {

unique_ptr<TensorProto> Serializer::Serialize(const Tensor &tensor,
                           const string &name) {
  MACE_NOT_IMPLEMENTED;
  return nullptr;
}

unique_ptr<Tensor> Serializer::Deserialize(const TensorProto &proto,
                                           DeviceType type) {
  unique_ptr<Tensor> tensor(new Tensor(GetDeviceAllocator(type),
                                       proto.data_type()));
  vector<index_t> dims;
  for (const index_t d : proto.dims()) {
    dims.push_back(d);
  }
  tensor->Resize(dims);

  switch (proto.data_type()) {
    case DT_FLOAT:
      tensor->Copy<float>(proto.float_data().data(),
                          proto.float_data().size());
      break;
    case DT_DOUBLE:
      tensor->Copy<double>(proto.double_data().data(),
                           proto.double_data().size());
      break;
    case DT_INT32:
      tensor->template Copy<int32_t>(proto.int32_data().data(),
                                   proto.int32_data().size());
      break;
    case DT_UINT8:
      tensor->CopyWithCast<int32_t, uint8_t>(proto.int32_data().data(),
                                         proto.int32_data().size());
      break;
    case DT_INT16:
      tensor->CopyWithCast<int32_t, int16_t>(proto.int32_data().data(),
                                         proto.int32_data().size());
      break;
    case DT_INT8:
      tensor->CopyWithCast<int32_t, int8_t>(proto.int32_data().data(),
                                        proto.int32_data().size());
      break;
    case DT_INT64:
      tensor->Copy<int64_t>(proto.int64_data().data(),
                          proto.int64_data().size());
      break;
    case DT_UINT16:
      tensor->CopyWithCast<int32_t, uint16_t>(proto.int32_data().data(),
                                          proto.int32_data().size());
      break;
    case DT_BOOL:
      tensor->CopyWithCast<int32_t, bool>(proto.int32_data().data(),
                                        proto.int32_data().size());
      break;
    case DT_STRING: {
      string *content = tensor->mutable_data<string>();
      for (int i = 0; i < proto.string_data().size(); ++i) {
        content[i] = proto.string_data(i);
      }
    }
      break;
    default:
      MACE_NOT_IMPLEMENTED;
      break;
  }

  return tensor;
}

} // namespace mace