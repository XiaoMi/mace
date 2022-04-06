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

#include "mace/core/proto/arg_helper.h"
#include "mace/core/proto/net_def_helper.h"
#include "mace/core/quantize.h"

namespace mace {

namespace {

template<typename T>
void DequantizeTensor(Runtime *runtime,
                      const unsigned char *model_data,
                      const ConstTensor &const_tensor,
                      Tensor *output_tensor) {
  Tensor::MappingGuard guard(output_tensor);
  auto quantized_data = reinterpret_cast<const uint8_t *>(
      model_data + const_tensor.offset());
  auto dequantized_data = output_tensor->mutable_data<T>();
  QuantizeUtil<T, uint8_t> quantize_util(&(runtime->thread_pool()));
  quantize_util.Dequantize(quantized_data,
                           output_tensor->size(),
                           const_tensor.scale(),
                           const_tensor.zero_point(),
                           dequantized_data);
}

}  // namespace

Workspace::Workspace(const OpDelegatorRegistry *registry, BaseFlow *flow) :
    op_delegator_registry_(registry),
    parent_flow_(flow) {}

const BaseFlow *Workspace::GetMaceFlow() const {
  return parent_flow_;
}

Tensor *Workspace::CreateTensor(const std::string &name, Runtime *runtime,
                                DataType dt, bool is_weight,
                                MemoryType mem_type,
                                BufferContentType content_type) {
  if (HasTensor(name)) {
    VLOG(3) << "Tensor " << name << " already exists. Skipping.";
  } else {
    VLOG(3) << "Creating Tensor " << name;
    if (mem_type == MemoryType::MEMORY_NONE) {
      mem_type = runtime->GetUsedMemoryType();
    }
    auto tensor = make_unique<Tensor>(runtime, dt, mem_type,
                                      std::vector<index_t>(),
                                      is_weight, name, content_type);
    tensor_map_[name] = std::move(tensor);
  }
  return GetTensor(name);
}

Workspace::~Workspace() {
  VLOG(1) << "Destroy Workspace";
}

Tensor *Workspace::GetTensor(const std::string &name) const {
  if (tensor_map_.count(name)) {
    return tensor_map_.at(name).get();
  } else {
    VLOG(1) << "Tensor " << name << " does not exist.";
  }
  return nullptr;
}

MaceStatus Workspace::AddTensor(const std::string &name,
                                std::unique_ptr<Tensor> tensor) {
  MACE_CHECK(tensor_map_.count(name) == 0, "tensor has exist: ", name);
  tensor_map_.emplace(name, std::move(tensor));
  return MaceStatus::MACE_SUCCESS;
}

std::vector<std::string> Workspace::Tensors() const {
  std::vector<std::string> names;
  for (auto &entry : tensor_map_) {
    names.push_back(entry.first);
  }
  return names;
}

MaceStatus Workspace::LoadModelTensor(const NetDef &net_def, Runtime *runtime,
                                      const unsigned char *model_data,
                                      const index_t model_data_size) {
  // When model has no weight, return immediately. Otherwise,
  // `MakeSliceBuffer` will try to map nullptr when running on GPU.
  if (model_data == nullptr && model_data_size == 0) {
    LOG(WARNING) << "Model has no weight, ignoring loading model tensor";
    return MaceStatus::MACE_SUCCESS;
  }
  MACE_CHECK(model_data != nullptr && model_data_size > 0);
  MACE_LATENCY_LOGGER(1, "Load model tensors");
  index_t valid_data_size = NetDefHelper::GetModelValidSize(net_def);
  VLOG(3) << "Model valid data size: " << valid_data_size;
  if (model_data_size >= 0) {
    MACE_CHECK(valid_data_size <= model_data_size,
               valid_data_size, " should be smaller than ", model_data_size);
  }

  const RuntimeType runtime_type = runtime->GetRuntimeType();
  auto slice_parent = runtime->MakeSliceBuffer(net_def, model_data,
                                               valid_data_size);
  diffused_buffer_ = (slice_parent == nullptr);
  if (diffused_buffer_) {
    bool is_quantize_model = NetDefHelper::IsQuantizedModel(net_def);
    for (const auto &const_tensor : net_def.tensors()) {
      MACE_LATENCY_LOGGER(2, "Load tensor ", const_tensor.name());
      VLOG(3) << "Tensor name: " << const_tensor.name()
              << ", data type: " << const_tensor.data_type() << ", shape: "
              << MakeString(std::vector<index_t>(const_tensor.dims().begin(),
                                                 const_tensor.dims().end()));
      std::vector<index_t> dims;
      for (const index_t d : const_tensor.dims()) {
        dims.push_back(d);
      }

      auto dst_data_type =
          runtime->GetComputeDataType(net_def, const_tensor);
      auto tensor = make_unique<Tensor>(
          runtime, dst_data_type, dims, true, const_tensor.name());
      runtime->AllocateBufferForTensor(tensor.get(), BufRentType::RENT_PRIVATE);

      const index_t tensor_end = const_tensor.offset() +
          tensor->size() * GetEnumTypeSize(const_tensor.data_type());
      MACE_CHECK(tensor_end <= model_data_size, "tensor_end (", tensor_end,
                 ") should <= ", model_data_size);

      if (runtime_type == RuntimeType::RT_CPU &&
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
        if (dst_data_type != DT_FLOAT) {
          DequantizeTensor<half>(runtime,
                                 model_data,
                                 const_tensor,
                                 tensor.get());
        } else {
          DequantizeTensor<float>(runtime,
                                  model_data,
                                  const_tensor,
                                  tensor.get());
        }
      } else {
        tensor->CopyBytes(model_data + const_tensor.offset(),
                          const_tensor.data_size() *
                              GetEnumTypeSize(const_tensor.data_type()));
      }

      tensor_map_[const_tensor.name()] = std::move(tensor);
    }
  } else {
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

      std::unique_ptr<Tensor> tensor = make_unique<Tensor>(
          runtime, const_tensor.data_type(), dims, true, const_tensor.name());
      tensor->SetScale(const_tensor.scale());
      tensor->SetZeroPoint(const_tensor.zero_point());
      MACE_CHECK_SUCCESS(runtime->AllocateBufferForTensor(
          tensor.get(), RENT_SLICE, slice_parent.get(), const_tensor.offset()));

      tensor_map_[const_tensor.name()] = std::move(tensor);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Workspace::AddQuantizeInfoForOutputTensor(
    const mace::NetDef &net_def, Runtime *runtime) {
  // add quantize info for output tensors.
  if (runtime->GetRuntimeType() == RuntimeType::RT_CPU) {
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
                                      Runtime *runtime) {
  std::unordered_set<std::string> tensor_to_host;
  for (auto &op : net_def.op()) {
    auto runtime_type = static_cast<RuntimeType>(op.device_type());
    if (runtime_type == RuntimeType::RT_CPU) {
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
            new Tensor(runtime, DataType::DT_FLOAT, dims,
                       true, const_tensor.name()));
        MACE_CHECK_SUCCESS(
            runtime->AllocateBufferForTensor(tensor.get(), RENT_PRIVATE));
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
            new Tensor(runtime, const_tensor.data_type(), GPU_BUFFER, dims,
                       true, const_tensor.name()));
        MACE_CHECK(tensor->size() == const_tensor.data_size(), tensor->name(),
                   " tensor's data_size not equal with the shape");
        MACE_CHECK_SUCCESS(runtime->AllocateBufferForTensor(
            tensor.get(), BufRentType::RENT_PRIVATE));
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

const OpDelegatorRegistry *Workspace::GetDelegatorRegistry() const {
  return op_delegator_registry_;
}

}  // namespace mace
