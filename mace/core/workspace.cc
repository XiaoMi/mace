// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#include "mace/core/memory_optimizer.h"
#include "mace/core/quantize.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_runtime.h"
#endif

namespace mace {

namespace {
bool HasQuantizedTensor(const NetDef &net_def) {
  for (auto &tensor : net_def.tensors()) {
    if (tensor.quantized()) {
      return true;
    }
  }
  return false;
}

bool HasHalfTensor(const NetDef &net_def) {
  for (auto &tensor : net_def.tensors()) {
    if (tensor.data_type() == DataType::DT_HALF) {
      return true;
    }
  }
  return false;
}

}  // namespace

Workspace::Workspace() = default;

Tensor *Workspace::CreateTensor(const std::string &name,
                                Allocator *alloc,
                                DataType type,
                                bool is_weight) {
  if (HasTensor(name)) {
    VLOG(3) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(3) << "Creating Tensor " << name;
    tensor_map_[name] = std::unique_ptr<Tensor>(new Tensor(alloc, type,
                                                           is_weight, name));
  }
  return GetTensor(name);
}

const Tensor *Workspace::GetTensor(const std::string &name) const {
  if (tensor_map_.count(name)) {
    return tensor_map_.at(name).get();
  } else {
    VLOG(1) << "Tensor " << name << " does not exist.";
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
    bool is_quantize_model = IsQuantizedModel(net_def);
    diffused_buffer_ =
        (device_type == DeviceType::CPU && HasHalfTensor(net_def)) ||
            (!is_quantize_model && HasQuantizedTensor(net_def));
#ifdef MACE_ENABLE_OPENCL
    diffused_buffer_ = diffused_buffer_ || (device_type == DeviceType::GPU &&
        device->gpu_runtime()->opencl_runtime()->GetDeviceMaxMemAllocSize() <=
            static_cast<uint64_t>(model_data_size));
#endif
    if (diffused_buffer_) {
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

        DataType dst_data_type = const_tensor.data_type();
        if ((device_type == DeviceType::CPU &&
             const_tensor.data_type() == DataType::DT_HALF) ||
            (!is_quantize_model && const_tensor.quantized())) {
          dst_data_type = DataType::DT_FLOAT;
        }

        std::unique_ptr<Tensor> tensor(
            new Tensor(device->allocator(), dst_data_type, true,
                       const_tensor.name()));
        tensor->Resize(dims);

        MACE_CHECK(tensor->size() == const_tensor.data_size(),
                   "Tensor's data_size not equal with the shape");
        MACE_CHECK(static_cast<index_t>(const_tensor.offset() +
            tensor->size() * GetEnumTypeSize(const_tensor.data_type())) <=
            model_data_size,
                   "buffer offset + length (",
                   const_tensor.offset(),
                   " + ",
                   tensor->size() * GetEnumTypeSize(const_tensor.data_type()),
                   ") should <= ",
                   model_data_size);

        if (device_type == DeviceType::CPU &&
            const_tensor.data_type() == DataType::DT_HALF) {
            // uncompress the weights of fp16
            auto org_data = reinterpret_cast<const half *>(
                model_data + const_tensor.offset());
            float *dst_data = tensor->mutable_data<float>();
            for (int i = 0; i < const_tensor.data_size(); ++i) {
              dst_data[i] = half_float::half_cast<float>(org_data[i]);
            }
        } else if (!is_quantize_model && const_tensor.quantized()) {
          // uncompress the weights of uint8
          Tensor::MappingGuard guard(tensor.get());
          auto quantized_data = reinterpret_cast<const uint8_t *>(
              model_data + const_tensor.offset());
          auto dequantized_data = tensor->mutable_data<float>();
          QuantizeUtil<uint8_t>
              quantize_util(&device->cpu_runtime()->thread_pool());
          quantize_util.Dequantize(quantized_data,
                                   tensor->size(),
                                   const_tensor.scale(),
                                   const_tensor.zero_point(),
                                   dequantized_data);
        } else {
          tensor->CopyBytes(model_data + const_tensor.offset(),
                            const_tensor.data_size() *
                                GetEnumTypeSize(const_tensor.data_type()));
        }

        tensor_map_[const_tensor.name()] = std::move(tensor);
      }
    } else {
      if (device_type == DeviceType::CPU) {
        tensor_buffer_ = std::unique_ptr<Buffer>(
            new Buffer(device->allocator(),
                       const_cast<unsigned char *>(model_data),
                       model_data_size));
      } else {
        tensor_buffer_ = std::unique_ptr<Buffer>(
            new Buffer(device->allocator()));
        MACE_RETURN_IF_ERROR(tensor_buffer_->Allocate(model_data_size));
        tensor_buffer_->Map(nullptr);
        tensor_buffer_->Copy(const_cast<unsigned char *>(model_data),
                             0, model_data_size);
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

        std::unique_ptr<Tensor> tensor(
            new Tensor(BufferSlice(
                tensor_buffer_.get(), const_tensor.offset(),
                const_tensor.data_size() *
                    GetEnumTypeSize(const_tensor.data_type())),
                       const_tensor.data_type(),
                       true,
                       const_tensor.name()));

        tensor->Reshape(dims);
        tensor->SetScale(const_tensor.scale());
        tensor->SetZeroPoint(const_tensor.zero_point());

        tensor_map_[const_tensor.name()] = std::move(tensor);
      }
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Workspace::PreallocateOutputTensor(
    const mace::NetDef &net_def,
    const mace::MemoryOptimizer *mem_optimizer,
    Device *device) {
  auto &mem_blocks = mem_optimizer->mem_blocks();
  for (auto &mem_block : mem_blocks) {
    VLOG(3) << "Preallocate memory block. id: " << mem_block.mem_id()
            << ", memory type: " << mem_block.mem_type()
            << ", size: " << mem_block.x() << "x" << mem_block.y();
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
          {static_cast<size_t>(mem_block.x()),
           static_cast<size_t>(mem_block.y())}, mem_block.data_type()));
      preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                        std::move(image_buf));
    } else if (mem_block.mem_type() == MemoryType::GPU_BUFFER) {
      std::unique_ptr<BufferBase> tensor_buf(
          new Buffer(device->allocator()));
      MACE_RETURN_IF_ERROR(tensor_buf->Allocate(
          mem_block.x() + MACE_EXTRA_BUFFER_PAD_SIZE));
      preallocated_allocator_.SetBuffer(mem_block.mem_id(),
                                        std::move(tensor_buf));
    }
  }
  VLOG(1) << "Preallocate buffer to tensors";
  for (auto &tensor_mem : mem_optimizer->tensor_mem_map()) {
    std::unique_ptr<Tensor> tensor
        (new Tensor(preallocated_allocator_.GetBuffer(tensor_mem.second.mem_id),
                    tensor_mem.second.data_type,
                    false, tensor_mem.first));
    tensor->set_data_format(tensor_mem.second.data_format);
    if (tensor_mem.second.data_format != DataFormat::NONE) {
      if (mem_blocks[tensor_mem.second.mem_id].mem_type()
          == MemoryType::GPU_IMAGE) {
        VLOG(1) << "Tensor: " << tensor_mem.first
                << " Mem: " << tensor_mem.second.mem_id
                << " Data type: " << tensor->dtype()
                << " Image shape: "
                << tensor->UnderlyingBuffer()->shape()[0]
                << ", "
                << tensor->UnderlyingBuffer()->shape()[1];
      } else {
        VLOG(1) << "Tensor: " << tensor_mem.first
                << " Mem: " << tensor_mem.second.mem_id
                << " Data type: " << tensor->dtype()
                << ", Buffer size: " << tensor->UnderlyingBuffer()->size();
      }
    }
    tensor_map_[tensor_mem.first] = std::move(tensor);
  }

  // add quantize info for output tensors.
  if (device->device_type() == DeviceType::CPU) {
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
  std::unordered_set<std::string> tensor_to_host;
  for (auto &op : net_def.op()) {
    if (op.device_type() == DeviceType::CPU) {
      for (std::string input : op.input()) {
        tensor_to_host.insert(input);
      }
    }
  }
  for (auto &const_tensor : net_def.tensors()) {
    auto iter = tensor_map_.find(const_tensor.name());
    if (iter->second->unused()) {
      tensor_map_.erase(iter);
    } else {
      std::vector<index_t> dims;
      for (const index_t d : const_tensor.dims()) {
        dims.push_back(d);
      }

      if (tensor_to_host.find(const_tensor.name()) != tensor_to_host.end()
          && const_tensor.data_type() == DataType::DT_HALF) {
        std::unique_ptr<Tensor> tensor(
            new Tensor(alloc, DataType::DT_FLOAT,
                       true, const_tensor.name()));
        tensor->Resize(dims);
        MACE_CHECK(tensor->size() == const_tensor.data_size(),
                   "Tensor's data_size not equal with the shape");
        Tensor::MappingGuard guard(tensor.get());
        float *dst_data = tensor->mutable_data<float>();
        const half *org_data = reinterpret_cast<const half *>(
            model_data + const_tensor.offset());
        for (index_t i = 0; i < const_tensor.data_size(); ++i) {
          dst_data[i] = half_float::half_cast<float>(org_data[i]);
        }
        tensor_map_[const_tensor.name()] = std::move(tensor);
      } else if (!diffused_buffer_) {
        std::unique_ptr<Tensor> tensor(
            new Tensor(alloc, const_tensor.data_type(),
                       true, const_tensor.name()));
        tensor->Resize(dims);
        MACE_CHECK(tensor->size() == const_tensor.data_size(),
                   "Tensor's data_size not equal with the shape");
        tensor->CopyBytes(model_data + const_tensor.offset(),
                          const_tensor.data_size() *
                              GetEnumTypeSize(const_tensor.data_type()));
        tensor_map_[const_tensor.name()] = std::move(tensor);
      }
    }
  }
  tensor_buffer_.reset(nullptr);
}

void Workspace::RemoveTensor(const std::string &name) {
  auto iter = tensor_map_.find(name);
  if (iter != tensor_map_.end()) {
    tensor_map_.erase(iter);
  }
}

}  // namespace mace
