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

#include "mace/core/workspace.h"

#include <unordered_set>
#include <utility>

#include "mace/core/arg_helper.h"
#include "mace/utils/quantize.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_runtime.h"
#endif

namespace mace {

namespace {
bool ShouldPreallocateMemoryForOp(const OperatorDef &op) {
  static const std::unordered_set<std::string> reuse_buffer_ops {
      "Reshape", "Identity", "Squeeze"
  };
  return reuse_buffer_ops.find(op.type()) == reuse_buffer_ops.end();
}

bool HasQuantizeOp(const NetDef &net_def) {
  for (auto &op : net_def.op()) {
    if (op.type() == "Quantize") {
      return true;
    }
  }
  return false;
}
}  // namespace

Workspace::Workspace() :
    host_scratch_buffer_(new ScratchBuffer(GetCPUAllocator())) {}

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
                                      Device *device,
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

  const DeviceType device_type = device->device_type();

  if (model_data_size > 0) {
#ifdef MACE_ENABLE_OPENCL
    if (device_type == DeviceType::GPU &&
        device->opencl_runtime()->GetDeviceMaxMemAllocSize() <=
            static_cast<uint64_t>(model_data_size)) {
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
            new Tensor(device->allocator(),
                       const_tensor.data_type(), true));
        tensor->Resize(dims);

        MACE_CHECK(tensor->size() == const_tensor.data_size(),
                   "Tensor's data_size not equal with the shape");
        MACE_CHECK(const_tensor.offset() + tensor->raw_size() <=
            model_data_size,
                   "buffer offset + length (",
                   const_tensor.offset(),
                   " + ",
                   tensor->raw_size(),
                   ") should <= ",
                   model_data_size);
        tensor->CopyBytes(model_data + const_tensor.offset(),
                          const_tensor.data_size() *
                              GetEnumTypeSize(const_tensor.data_type()));

        tensor_map_[const_tensor.name()] = std::move(tensor);
      }
      fused_buffer_ = false;
    } else {
#else
    {
#endif
      if (device_type == DeviceType::CPU) {
        tensor_buffer_ = std::unique_ptr<Buffer>(
            new Buffer(device->allocator(),
                       const_cast<unsigned char*>(model_data),
                       model_data_size));
      } else {
        tensor_buffer_ = std::unique_ptr<Buffer>(
            new Buffer(device->allocator()));
        MACE_RETURN_IF_ERROR(tensor_buffer_->Allocate(model_data_size));
        tensor_buffer_->Map(nullptr);
        tensor_buffer_->Copy(const_cast<unsigned char*>(model_data),
                             0, model_data_size);
        tensor_buffer_->UnMap();
      }
      bool has_quantize_op = HasQuantizeOp(net_def);
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
            new Tensor(BufferSlice(
                tensor_buffer_.get(), const_tensor.offset(),
                const_tensor.data_size() *
                    GetEnumTypeSize(const_tensor.data_type())),
                       const_tensor.data_type(), true));
        tensor->Reshape(dims);
        tensor->SetScale(const_tensor.scale());
        tensor->SetZeroPoint(const_tensor.zero_point());

        // Only weights are quantized
        if (const_tensor.quantized() && !has_quantize_op) {
          std::unique_ptr<Tensor> dequantized_tensor(new Tensor(true));
          dequantized_tensor->Resize(dims);
          Tensor::MappingGuard quantize_guard(tensor.get());
          Tensor::MappingGuard dequantize_guard(dequantized_tensor.get());
          auto quantized_data = tensor->data<uint8_t>();
          auto dequantized_data = dequantized_tensor->mutable_data<float>();
          Dequantize(quantized_data,
                     tensor->size(),
                     tensor->scale(),
                     tensor->zero_point(),
                     dequantized_data);
          tensor_map_[const_tensor.name()] = std::move(dequantized_tensor);
        } else {
          tensor_map_[const_tensor.name()] = std::move(tensor);
        }
      }
      fused_buffer_ = true;
    }
  }

  if (device_type == DeviceType::CPU || device_type == DeviceType::GPU) {
    MaceStatus status = CreateOutputTensorBuffer(net_def, device);
    if (status != MaceStatus::MACE_SUCCESS) return status;
  }

  if (device_type == DeviceType::CPU) {
    for (const auto &op : net_def.op()) {
      VLOG(2) << "Add quantize info for op: " << op.name();
      MACE_CHECK(op.quantize_info().empty()
                     || op.quantize_info().size() == op.output().size(),
                 "quantize info size must be equal to output size or empty");
      for (int i = 0; i < op.quantize_info().size(); ++i) {
        auto &quantize_info = op.quantize_info(i);
        Tensor *tensor = GetTensor(op.output(i));
        tensor->SetScale(quantize_info.scale());
        tensor->SetZeroPoint(quantize_info.zero_point());
        tensor->SetMinVal(quantize_info.minval());
        tensor->SetMaxVal(quantize_info.maxval());
      }
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Workspace::CreateOutputTensorBuffer(const NetDef &net_def,
                                               Device *device) {
  DeviceType device_type = device->device_type();
  DataType dtype = DataType::DT_INVALID;
  if (net_def.mem_arena().mem_block_size() > 0) {
    // We use the data type of the first op with mem id,
    // as CPU&GPU have consistent data type for each layer for now.
    // As DSP may have different data output type for each op,
    // we stick to the same concept.
    for (auto &op : net_def.op()) {
      // TODO(liuqi): refactor to add device_type to OperatorDef
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
  }
  // TODO(liyin): memory block should not have concept of type, but to be
  // consistent with gpu, all memory block use float/half as unit
  for (auto &mem_block : net_def.mem_arena().mem_block()) {
    if (mem_block.device_type() == device_type) {
      VLOG(3) << "Preallocate memory block. id: " << mem_block.mem_id()
              << ", device type: " << mem_block.device_type()
              << ", memory type: " << mem_block.mem_type();
      if (mem_block.mem_type() == MemoryType::CPU_BUFFER) {
        std::unique_ptr<BufferBase> tensor_buf(
            new Buffer(GetCPUAllocator()));
        MACE_RETURN_IF_ERROR(tensor_buf->Allocate(
            mem_block.x() + MACE_EXTRA_BUFFER_PAD_SIZE));
        preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                          std::move(tensor_buf));
      } else if (mem_block.mem_type() == MemoryType::GPU_IMAGE) {
        std::unique_ptr<BufferBase> image_buf(
            new Image(device->allocator()));
        MACE_RETURN_IF_ERROR(image_buf->Allocate(
            {mem_block.x(), mem_block.y()}, dtype));
        preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                          std::move(image_buf));
      } else if (mem_block.mem_type() == MemoryType::GPU_BUFFER) {
        std::unique_ptr<BufferBase> tensor_buf(
            new Buffer(device->allocator()));
        MACE_RETURN_IF_ERROR(tensor_buf->Allocate(
            mem_block.x() * GetEnumTypeSize(dtype)));
        preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                          std::move(tensor_buf));
      }
    }
  }
  VLOG(3) << "Preallocate buffer to tensors";
  for (auto &op : net_def.op()) {
    // TODO(liuqi): refactor to add device_type to OperatorDef
    const int op_device =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            op, "device", static_cast<int>(device_type));
    if (op_device == device_type) {
      if (!op.mem_id().empty()
          && ShouldPreallocateMemoryForOp(op)) {
        auto mem_ids = op.mem_id();
        int count = mem_ids.size();
        for (int i = 0; i < count; ++i) {
          DataType output_type;
          if (i < op.output_type_size()) {
            output_type = op.output_type(i);
          } else {
            output_type = dtype;
          }
          std::unique_ptr<Tensor> tensor
              (new Tensor(preallocated_allocator_.GetBuffer(mem_ids[i]),
                          output_type));
          tensor->SetSourceOpName(op.name());
          if (device_type == DeviceType::GPU) {
            VLOG(3) << "Tensor: " << op.name() << "(" << op.type() << ")"
                    << " Mem: " << mem_ids[i]
                    << " Image shape: "
                    << dynamic_cast<Image *>(tensor->UnderlyingBuffer())
                        ->image_shape()[0]
                    << ", "
                    << dynamic_cast<Image *>(tensor->UnderlyingBuffer())
                        ->image_shape()[1];
          } else if (device_type == DeviceType::CPU) {
            VLOG(3) << "Tensor: " << op.name() << "(" << op.type() << ")"
                    << " Mem: " << mem_ids[i]
                    << ", Buffer size: " << tensor->UnderlyingBuffer()->size();
          }
          tensor_map_[op.output(i)] = std::move(tensor);
        }
      } else {
        for (int i = 0; i < op.output().size(); ++i) {
          MACE_CHECK(
              op.output_type_size() == 0
                  || op.output_size()
                      == op.output_type_size(),
              "operator output size != operator output type size",
              op.output_size(),
              op.output_type_size());
          DataType output_type;
          if (i < op.output_type_size()) {
            output_type = op.output_type(i);
          } else {
            output_type = static_cast<DataType>(ProtoArgHelper::GetOptionalArg(
                op, "T", static_cast<int>(DT_FLOAT)));
          }
          CreateTensor(op.output(i),
                       device->allocator(),
                       output_type);
        }
      }

      for (int output_idx = 0; output_idx < op.output_shape_size();
           ++output_idx) {
        std::vector<index_t>
            shape_configured(op.output_shape(output_idx).dims_size());
        for (size_t dim = 0; dim < shape_configured.size(); ++dim) {
          shape_configured[dim] = op.output_shape(output_idx).dims(dim);
        }
        tensor_map_[op.output(output_idx)]->SetShapeConfigured(
            shape_configured);
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

void Workspace::RemoveUnusedBuffer() {
  auto iter = tensor_map_.begin();
  auto end_iter = tensor_map_.end();
  while (iter != end_iter) {
    auto old_iter = iter++;
    if (old_iter->second->unused()) {
      tensor_map_.erase(old_iter);
    }
  }
  tensor_buffer_.reset(nullptr);
}

void Workspace::RemoveAndReloadBuffer(const NetDef &net_def,
                                      const unsigned char *model_data,
                                      Allocator *alloc) {
  for (auto &const_tensor : net_def.tensors()) {
    auto iter = tensor_map_.find(const_tensor.name());
    if (iter->second->unused()) {
      tensor_map_.erase(iter);
    } else if (fused_buffer_) {
      tensor_map_.erase(iter);
      std::vector<index_t> dims;
      for (const index_t d : const_tensor.dims()) {
        dims.push_back(d);
      }
      std::unique_ptr<Tensor> tensor(
          new Tensor(alloc, const_tensor.data_type()));
      tensor->Resize(dims);
      MACE_CHECK(tensor->size() == const_tensor.data_size(),
                 "Tensor's data_size not equal with the shape");
      tensor->CopyBytes(model_data + const_tensor.offset(),
                        const_tensor.data_size() *
                            GetEnumTypeSize(const_tensor.data_type()));

      tensor_map_[const_tensor.name()] = std::move(tensor);
    }
  }
  tensor_buffer_.reset(nullptr);
}

}  // namespace mace
