// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <string>
#include <vector>
#include <unordered_set>
#include <utility>

#include "mace/core/arg_helper.h"
#include "mace/core/workspace.h"
#include "mace/utils/timer.h"

namespace mace {

namespace {
bool ShouldPreallocateMemoryForOp(const OperatorDef &op) {
  static const std::unordered_set<std::string> reuse_buffer_ops {
      "Reshape", "Identity", "Squeeze"
  };
  return reuse_buffer_ops.find(op.type()) == reuse_buffer_ops.end();
}
}  // namespace

Workspace::Workspace() : host_scratch_buffer_(new ScratchBuffer(
  GetDeviceAllocator(DeviceType::CPU))) {}

Tensor *Workspace::CreateTensor(const std::string &name,
                                Allocator *alloc,
                                DataType type) {
  if (HasTensor(name)) {
    VLOG(3) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(3) << "Creating Tensor " << name;
    tensor_map_[name] = std::unique_ptr<Tensor>(new Tensor(alloc, type));
    tensor_map_[name]->SetSourceOpName(name);
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

MaceStatus Workspace::LoadModelTensor(const NetDef &net_def,
                                      DeviceType type,
                                      const unsigned char *model_data) {
  MACE_LATENCY_LOGGER(1, "Load model tensors");
  index_t model_data_size = 0;
  for (auto &const_tensor : net_def.tensors()) {
    model_data_size = std::max(
        model_data_size,
        static_cast<index_t>(const_tensor.offset() +
                             const_tensor.data_size() *
                             GetEnumTypeSize(const_tensor.data_type())));
  }
  VLOG(3) << "Model data size: " << model_data_size;

  if (model_data_size > 0) {
    if (type == DeviceType::CPU) {
      tensor_buffer_ = std::unique_ptr<Buffer>(
          new Buffer(GetDeviceAllocator(type),
                     const_cast<unsigned char*>(model_data),
                     model_data_size));
    } else {
      tensor_buffer_ = std::unique_ptr<Buffer>(
          new Buffer(GetDeviceAllocator(type)));
      MACE_RETURN_IF_ERROR(tensor_buffer_->Allocate(model_data_size));
      tensor_buffer_->Map(nullptr);
      tensor_buffer_->Copy(const_cast<unsigned char*>(model_data),
                           0, model_data_size);
      tensor_buffer_->UnMap();
    }
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

    std::unique_ptr<Tensor> tensor(
        new Tensor(BufferSlice(tensor_buffer_.get(), const_tensor.offset(),
                               const_tensor.data_size() *
                                   GetEnumTypeSize(const_tensor.data_type())),
                   const_tensor.data_type()));

    tensor->Reshape(dims);
    tensor_map_[const_tensor.name()] = std::move(tensor);
  }

  if (type == DeviceType::CPU || type == DeviceType::GPU) {
    MaceStatus status = CreateOutputTensorBuffer(net_def, type);
    if (status != MaceStatus::MACE_SUCCESS) return status;
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Workspace::CreateOutputTensorBuffer(const NetDef &net_def,
                                               DeviceType device_type) {
  if (!net_def.has_mem_arena() || net_def.mem_arena().mem_block_size() == 0) {
    return MaceStatus::MACE_SUCCESS;
  }

  DataType dtype = DataType::DT_INVALID;
  // We use the data type of the first op with mem id,
  // as CPU&GPU have consistent data type for each layer for now.
  // As DSP may have different data output type for each op,
  // we stick to the same concept.
  for (auto &op : net_def.op()) {
    // TODO(liuqi): refactor based on PB
    const int op_device =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "device", static_cast<int>(device_type));
    if (op_device == device_type && !op.mem_id().empty()) {
      const DataType op_dtype = static_cast<DataType>(
          ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
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
    if (device_type == DeviceType::GPU) {
      // TODO(liuqi): refactor based on PB
      if (mem_block.mem_id() >= 20000) {
        std::unique_ptr<BufferBase> image_buf(
            new Image());
        MACE_RETURN_IF_ERROR(image_buf->Allocate(
            {mem_block.x(), mem_block.y()}, dtype));
        preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                          std::move(image_buf));
      }
    } else {
      if (mem_block.mem_id() < 20000) {
        std::unique_ptr<BufferBase> tensor_buf(
            new Buffer(GetDeviceAllocator(device_type)));
        MACE_RETURN_IF_ERROR(tensor_buf->Allocate(
            mem_block.x() * GetEnumTypeSize(dtype)
            + MACE_EXTRA_BUFFER_PAD_SIZE));
        preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                          std::move(tensor_buf));
      }
    }
  }
  VLOG(3) << "Preallocate buffer to tensors";
  for (auto &op : net_def.op()) {
    // TODO(liuqi): refactor based on PB
    const int op_device =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "device", static_cast<int>(device_type));
    if (op_device == device_type && !op.mem_id().empty()
        && ShouldPreallocateMemoryForOp(op)) {
      auto mem_ids = op.mem_id();
      int count = mem_ids.size();
      for (int i = 0; i < count; ++i) {
        std::unique_ptr<Tensor> tensor
            (new Tensor(preallocated_allocator_.GetBuffer(mem_ids[i]), dtype));
        tensor->SetSourceOpName(op.name());
        if (device_type == DeviceType::GPU) {
          VLOG(3) << "Tensor: " << op.name() << "(" << op.type() << ")"
                  << " Mem: "  << mem_ids[i]
                  << " Image shape: "
                  << dynamic_cast<Image *>(tensor->UnderlyingBuffer())
                      ->image_shape()[0]
                  << ", "
                  << dynamic_cast<Image *>(tensor->UnderlyingBuffer())
                      ->image_shape()[1];
        } else if (device_type == DeviceType::CPU) {
          VLOG(3) << "Tensor: " << op.name() << "(" << op.type() << ")"
                  << " Mem: "  << mem_ids[i]
                  << ", Buffer size: " << tensor->UnderlyingBuffer()->size();
        }
        tensor_map_[op.output(i)] = std::move(tensor);
      }
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

ScratchBuffer *Workspace::GetScratchBuffer(DeviceType device_type) {
  if (device_type == CPU) {
    return host_scratch_buffer_.get();
  } else {
    return nullptr;
  }
}

}  // namespace mace
