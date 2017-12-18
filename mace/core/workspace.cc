//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/workspace.h"
#include "mace/core/serializer.h"
#include "mace/core/arg_helper.h"

namespace mace {

vector<string> Workspace::Tensors() const {
  vector<string> names;
  for (auto &entry : tensor_map_) {
    names.push_back(entry.first);
  }
  return names;
}

Tensor *Workspace::CreateTensor(const string &name,
                                Allocator *alloc,
                                DataType type) {
  if (HasTensor(name)) {
    VLOG(1) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(1) << "Creating Tensor " << name;
    tensor_map_[name] = unique_ptr<Tensor>(new Tensor(alloc, type));
  }
  return GetTensor(name);
}

bool Workspace::RemoveTensor(const string &name) {
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    VLOG(1) << "Removing blob " << name << " from this workspace.";
    tensor_map_.erase(it);
    return true;
  }
  return false;
}

const Tensor *Workspace::GetTensor(const string &name) const {
  if (tensor_map_.count(name)) {
    return tensor_map_.at(name).get();
  } else {
    LOG(WARNING) << "Tensor " << name << " does not exist.";
  }
  return nullptr;
}

Tensor *Workspace::GetTensor(const string &name) {
  return const_cast<Tensor *>(
      static_cast<const Workspace *>(this)->GetTensor(name));
}

void Workspace::LoadModelTensor(const NetDef &net_def, DeviceType type) {
  Serializer serializer;
  for (auto &tensor_proto : net_def.tensors()) {
    VLOG(1) << "Load tensor: " << tensor_proto.name()
            << ", with data type: " << tensor_proto.data_type()
            << ", has shape: "
            << internal::MakeString(vector<index_t>(tensor_proto.dims().begin(),
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
  std::map<std::string, std::shared_ptr<Tensor>> mem_tensor_map;
  const DataType dtype = static_cast<DataType>(
      ArgumentHelper::GetSingleArgument<OperatorDef, int>(
          net_def.op(0),
          "T",
          static_cast<int>(DT_FLOAT)));
  for (auto &mem_block: net_def.mem_arena().mem_block()) {
    string mem_block_name = MemBlockName(mem_block.mem_id());
    mem_tensor_map[mem_block_name].reset(new Tensor(
        GetDeviceAllocator(DeviceType::OPENCL),
        dtype));
    mem_tensor_map[mem_block_name]->AllocateImageMemory({mem_block.x(),
                                                         mem_block.y()});
  }
  for (auto &op: net_def.op()) {
    if (op.has_mem_id()) {
      tensor_map_[op.output(0)] = mem_tensor_map[MemBlockName(op.mem_id())];
    }
  }
}

}  // namespace mace