//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include <vector>

#include "mace/core/arg_helper.h"
#include "mace/core/workspace.h"
#include "mace/utils/timer.h"

namespace mace {

Tensor *Workspace::CreateTensor(const std::string &name,
                                Allocator *alloc,
                                DataType type) {
  if (HasTensor(name)) {
    VLOG(3) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(3) << "Creating Tensor " << name;
    tensor_map_[name] =
        std::move(std::unique_ptr<Tensor>(new Tensor(alloc, type)));
  }
  return GetTensor(name);
}

const Tensor *Workspace::GetTensor(const std::string &name) const {
  if (tensor_map_.count(name)) {
    return tensor_map_.at(name).get();
  } else {
    LOG(WARNING) << "Tensor " << name << " does not exist.";
  }
  return nullptr;
}

Tensor *Workspace::GetTensor(const std::string &name) {
  return const_cast<Tensor *>(
      static_cast<const Workspace *>(this)->GetTensor(name));
}

std::vector<std::string> Workspace::Tensors() const {
  std::vector<std::string> names;
  for (auto &entry : tensor_map_) {
    names.push_back(entry.first);
  }
  return names;
}

void Workspace::LoadModelTensor(const NetDef &net_def, DeviceType type) {
  MACE_LATENCY_LOGGER(1, "Load model tensors");
  index_t model_data_size = 0;
  unsigned char *model_data_ptr = nullptr;
  for (auto &const_tensor : net_def.tensors()) {
    if (model_data_ptr == nullptr ||
        reinterpret_cast<long long>(const_tensor.data()) <
            reinterpret_cast<long long>(model_data_ptr)) {
      model_data_ptr = const_cast<unsigned char *>(const_tensor.data());
    }
  }
  for (auto &const_tensor : net_def.tensors()) {
    model_data_size = std::max(
        model_data_size,
        static_cast<index_t>((reinterpret_cast<long long>(const_tensor.data()) -
                              reinterpret_cast<long long>(model_data_ptr)) +
                             const_tensor.data_size() *
                                 GetEnumTypeSize(const_tensor.data_type())));
  }
  VLOG(3) << "Model data size: " << model_data_size;

  if (type == DeviceType::CPU) {
    tensor_buffer_ = std::move(std::unique_ptr<Buffer>(
        new Buffer(GetDeviceAllocator(type), model_data_ptr, model_data_size)));
  } else {
    tensor_buffer_ = std::move(std::unique_ptr<Buffer>(
        new Buffer(GetDeviceAllocator(type), model_data_size)));
    tensor_buffer_->Map(nullptr);
    tensor_buffer_->Copy(model_data_ptr, 0, model_data_size);
    tensor_buffer_->UnMap();
  }

  for (auto &const_tensor : net_def.tensors()) {
    MACE_LATENCY_LOGGER(2, "Load tensor ", const_tensor.name());
    VLOG(3) << "Tensor name: " << const_tensor.name()
            << ", data type: " << const_tensor.data_type() << ", shape: "
            << MakeString(std::vector<index_t>(const_tensor.dims().begin(),
                                               const_tensor.dims().end()));
    std::vector<index_t> dims;
    for (const index_t d : const_tensor.dims()) {
      dims.push_back(d);
    }

    index_t offset = (long long)const_tensor.data() - (long long)model_data_ptr;
    std::unique_ptr<Tensor> tensor(
        new Tensor(BufferSlice(tensor_buffer_.get(), offset,
                               const_tensor.data_size() *
                                   GetEnumTypeSize(const_tensor.data_type())),
                   const_tensor.data_type()));

    tensor->Reshape(dims);
    tensor_map_[const_tensor.name()] = std::move(tensor);
  }

  if (type == DeviceType::OPENCL) {
    CreateImageOutputTensor(net_def);
  }
}

void Workspace::CreateImageOutputTensor(const NetDef &net_def) {
  if (!net_def.has_mem_arena() || net_def.mem_arena().mem_block_size() == 0) {
    return;
  }

  DataType dtype = DataType::DT_INVALID;
  // We use the data type of the first op (with mem id, must be image),
  // as GPU have consistent data type for each layer for now.
  // As DSP may have different data output type for each op,
  // we stick to the same concept.
  for (auto &op : net_def.op()) {
    if (! op.mem_id().empty()){
      const DataType op_dtype = static_cast<DataType>(
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(
              op, "T", static_cast<int>(DT_FLOAT)));
      if (op_dtype != DataType::DT_INVALID) {
        dtype = op_dtype;
        // find first valid data type, break
        break;
      }
    }
  }
  MACE_CHECK(dtype != DataType::DT_INVALID, "data type is invalid.");
  for (auto &mem_block : net_def.mem_arena().mem_block()) {
    std::unique_ptr<BufferBase> image_buf(
        new Image({mem_block.x(), mem_block.y()}, dtype));
    preallocated_allocator_.SetBuffer(mem_block.mem_id(), std::move(image_buf));
  }
  VLOG(3) << "Preallocate image to tensors";
  for (auto &op : net_def.op()) {
    if (!op.mem_id().empty()) {
      auto mem_ids = op.mem_id();
      int count = mem_ids.size();
      for (int i = 0; i < count; ++i) {
        std::unique_ptr<Tensor> tensor
            (new Tensor(preallocated_allocator_.GetBuffer(mem_ids[i]), dtype));
        tensor->SetSourceOpName(op.name());
        VLOG(3) << "Tensor: " << op.name() << "(" << op.type() << ")" << "; Mem: "
                << mem_ids[i] << "; Image shape: "
                << dynamic_cast<Image *>(tensor->UnderlyingBuffer())->image_shape()[0]
                << ", "
                << dynamic_cast<Image *>(tensor->UnderlyingBuffer())->image_shape()[1];
        tensor_map_[op.output(i)] = std::move(tensor);
      }
    }
  }
}

}  // namespace mace
