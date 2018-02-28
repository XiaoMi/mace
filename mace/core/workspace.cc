//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include <vector>

#include "mace/core/workspace.h"
#include "mace/core/serializer.h"
#include "mace/core/arg_helper.h"
#include "mace/core/runtime/opencl/opencl_preallocated_pooled_allocator.h"
#include "mace/utils/timer.h"

namespace mace {

std::vector<std::string> Workspace::Tensors() const {
  std::vector<std::string> names;
  for (auto &entry : tensor_map_) {
    names.push_back(entry.first);
  }
  return names;
}

Tensor *Workspace::CreateTensor(const std::string &name,
                                Allocator *alloc,
                                DataType type) {
  if (HasTensor(name)) {
    VLOG(3) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(3) << "Creating Tensor " << name;
    tensor_map_[name] = std::move(std::unique_ptr<Tensor>(new Tensor(alloc, type)));
  }
  return GetTensor(name);
}

bool Workspace::RemoveTensor(const std::string &name) {
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    VLOG(3) << "Removing blob " << name << " from this workspace.";
    tensor_map_.erase(it);
    return true;
  }
  return false;
}

const Tensor *Workspace::GetTensor(const std::string &name) const {
  if (tensor_map_.count(name)) {
    return tensor_map_.at(name).get();
  } else {
    LOG(WARNING) << "Tensor " << name << " does not exist.";
  }
  return nullptr;
}


void Workspace::RemoveUnsedTensor() {
  auto iter = tensor_map_.begin();
  auto end_iter = tensor_map_.end();
  while(iter != end_iter) {
    auto old_iter = iter++;
    if(old_iter->second->unused()) {
      tensor_map_.erase(old_iter);
    }
  }
}

Tensor *Workspace::GetTensor(const std::string &name) {
  return const_cast<Tensor *>(
      static_cast<const Workspace *>(this)->GetTensor(name));
}

void Workspace::LoadModelTensor(const NetDef &net_def, DeviceType type) {
  MACE_LATENCY_LOGGER(1, "Load model tensors");
  Serializer serializer;
  for (auto &tensor_proto : net_def.tensors()) {
    MACE_LATENCY_LOGGER(2, "Load tensor ", tensor_proto.name());
    VLOG(3) << "Tensor name: " << tensor_proto.name()
            << ", data type: " << tensor_proto.data_type()
            << ", shape: "
            << MakeString(std::vector<index_t>(tensor_proto.dims().begin(),
                                               tensor_proto.dims().end()));
    tensor_map_[tensor_proto.name()] =
        serializer.Deserialize(tensor_proto, type);
  }
  if (type == DeviceType::OPENCL) {
    CreateImageOutputTensor(net_def);
  }
}

void Workspace::CreateImageOutputTensor(const NetDef &net_def) {
  if (!net_def.has_mem_arena() || net_def.mem_arena().mem_block_size() == 0) {
    return;
  }
  preallocated_allocator_ =
    std::move(std::unique_ptr<PreallocatedPooledAllocator>(
      new OpenCLPreallocatedPooledAllocator));

  DataType dtype = DataType::DT_INVALID;
  // We use the data type of the first op (with mem id, must be image),
  // as GPU have consistent data type for each layer for now.
  // As DSP may have different data output type for each op,
  // we stick to the same concept.
  for (auto &op: net_def.op()) {
    if (op.has_mem_id()) {
      const DataType op_dtype = static_cast<DataType>(
        ArgumentHelper::GetSingleArgument<OperatorDef, int>(
          op,
          "T",
          static_cast<int>(DT_FLOAT)));
      if (op_dtype != DataType::DT_INVALID) {
        dtype = op_dtype;
        // find first valid data type, break
        break;
      }
    }
  }
  MACE_CHECK(dtype != DataType::DT_INVALID, "data type is invalid.");
  for (auto &mem_block: net_def.mem_arena().mem_block()) {
    preallocated_allocator_->PreallocateImage(mem_block.mem_id(),
                                              {mem_block.x(), mem_block.y()},
                                              dtype);
  }
  VLOG(3) << "Preallocate image to tensors";
  auto allocator = GetDeviceAllocator(DeviceType::OPENCL);
  for (auto &op: net_def.op()) {
    if (op.has_mem_id()) {
      CreateTensor(op.output(0), allocator, dtype);
      tensor_map_[op.output(0)]->PreallocateImage(
        preallocated_allocator_->GetImage(op.mem_id()),
        preallocated_allocator_->GetImageSize(op.mem_id()));
    }
  }
}

}  // namespace mace
