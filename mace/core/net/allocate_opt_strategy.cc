// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/core/net/allocate_strategy.h"

#include <list>

#include "mace/core/tensor.h"
#include "mace/utils/logging.h"

namespace mace {

namespace {
typedef std::list<std::unique_ptr<Buffer>> BufferList;

struct TensorRef {
  Tensor *tensor;
  int refs;
  Buffer *buffer;

  explicit TensorRef(Tensor *tensor_ptr)
      : tensor(tensor_ptr), refs(1), buffer(nullptr) {}
};

// If *monotonous return false, the compare result is meaningless
int CompareShape(const std::vector<index_t> &shape1,
                 const std::vector<index_t> &shape2, bool *monotonous) {
  auto shape_size = shape1.size();
  MACE_CHECK(shape_size == shape2.size());

  index_t area1 = 1;
  index_t area2 = 1;
  *monotonous = true;
  int sum_plus = 0;
  for (size_t i = 0; i < shape_size; ++i) {
    index_t interval = shape1[i] - shape2[i];
#ifdef MACE_DISABLE_BIG_BLOCK
    // This code ensures the allocated buffers no bigger than the max shape
    if ((interval >= 0 && sum_plus < 0) || (interval <= 0 && sum_plus > 0)) {
      *monotonous = false;
      break;
    }
#endif
    sum_plus += interval;
    area1 *= shape1[i];
    area2 *= shape2[i];
  }

  return area1 - area2;
}

BufferList::iterator FindBestFreeBuffer(
    const MemInfo &mem_info,
    BufferList *free_buf_list, bool *need_expand) {
  index_t best_waste_area = LLONG_MAX;
  index_t best_lack_area = LLONG_MIN;
  bool find_free = false;
  BufferList::iterator best_idx = free_buf_list->end();
  for (auto i = free_buf_list->begin(); i != free_buf_list->end(); ++i) {
    if ((*i)->mem_type != mem_info.mem_type ||
        (*i)->data_type != mem_info.data_type) {
      continue;
    }

    bool monotonous = false;
    int compare = CompareShape((*i)->dims, mem_info.dims, &monotonous);
    if (!monotonous) {
      continue;
    }

    if (compare < 0) {
      if (!find_free || compare > best_lack_area) {
        best_lack_area = compare;
        *need_expand = true;
        best_idx = i;
      } else {
        continue;
      }
    } else {
      if (compare < best_waste_area) {
        best_waste_area = compare;
        find_free = true;
        *need_expand = false;
        best_idx = i;
      } else {
        continue;
      }
    }
  }

  return best_idx;
}

void SimulateAllocateBuffer(std::shared_ptr<TensorRef> tensor_ref,
                            BufferList *used_buf_list,
                            BufferList *free_buf_list) {
  const Tensor *tensor = tensor_ref->tensor;
  Runtime *runtime = tensor->GetCurRuntime();
  BufferContentType content_type = BufferContentType::IN_OUT_CHANNEL;
  unsigned int content_param = 0;
  tensor->GetContentType(&content_type, &content_param);
  auto mem_type = tensor->memory_type();
  auto data_type = tensor->dtype();
  auto &tensor_dims = tensor->shape();
  std::vector<index_t> buf_dims = runtime->ComputeBufDimFromTensorDim(
      tensor_dims, mem_type, content_type, content_param);
  bool need_expand = false;
  MemInfo buf_info(mem_type, data_type, buf_dims);
  auto idx = FindBestFreeBuffer(buf_info, free_buf_list, &need_expand);

  std::unique_ptr<Buffer> buffer;
  if (idx == free_buf_list->end()) {
    buffer = make_unique<Buffer>(mem_type, data_type, buf_dims);
  } else {
    buffer = std::move(*idx);
    free_buf_list->erase(idx);
    if (need_expand) {
      buffer->dims = buf_dims;
    }
  }
  VLOG(3) << "tensor name: " << tensor_ref->tensor->name()
          << ", tensor shape: " << MakeString(tensor_ref->tensor->shape())
          << ", buffer shape: " << MakeString(buffer->dims)
          << ", set ptr: " << buffer.get();
  tensor_ref->buffer = buffer.get();
  used_buf_list->push_back(std::move(buffer));
}

void SimulateDeleteBuffer(std::shared_ptr<TensorRef> tensor_ref,
                          BufferList *used_buf_list,
                          BufferList *free_buf_list) {
  Buffer *buffer = tensor_ref->buffer;
  BufferList::iterator idx = used_buf_list->end();
  for (auto i = used_buf_list->begin(); i != used_buf_list->end(); ++i) {
    if (i->get() == buffer) {
      idx = i;
      break;
    }
  }
  MACE_CHECK(idx != used_buf_list->end(),
             "Can not find used buffer: ",
             tensor_ref->tensor->name(),
             buffer);
  free_buf_list->push_back(std::move(*idx));
  used_buf_list->erase(idx);
}

void ReallyAllocateBuffer(
    std::unordered_map<std::string, std::shared_ptr<TensorRef>> tensor_refs) {
  for (auto i = tensor_refs.begin(); i != tensor_refs.end(); ++i) {
    Buffer *buffer = i->second->buffer;
    if (buffer == nullptr) {
      VLOG(3) << "ReallyAllocateBuffer, tensor " << i->second->tensor->name()
              << " is model's input";
      continue;
    }
    Runtime *runtime = i->second->tensor->GetCurRuntime();
    if (buffer->memory<void>() == nullptr) {
      auto new_buf = runtime->ObtainBuffer(*buffer, RENT_SHARE);
      buffer->SetBuf(new_buf->mutable_memory<void>());
      VLOG(3) << "ReallyAllocateBuffer, allocate: " << buffer->memory<void>()
              << ", buffer dim is: " << MakeString(buffer->dims)
              << ", the buffer is: " << buffer
              << ", final refs: " << i->second->refs;
    }
    Tensor *tensor = i->second->tensor;
    runtime->SetBufferToTensor(make_unique<Buffer>(*buffer), tensor);
  }
}
}  // namespace

template<>
MaceStatus AllocateTensorMemory<SERIAL_OPT>(const OperationArray &operators) {
  std::unordered_map<std::string, std::shared_ptr<TensorRef>> tensor_refs;
  // Collect the refs of input tensor
  for (auto &op : operators) {
    size_t input_size = static_cast<size_t>(op->InputSize());
    for (size_t i = 0; i < input_size; ++i) {
      const Tensor *tensor = op->Input(i);
      if (tensor->is_weight()) {
        continue;
      }
      auto tensor_name = tensor->name();
      if (tensor_refs.count(tensor_name) == 0) {
        tensor_refs.emplace(tensor_name, std::make_shared<TensorRef>(
            const_cast<Tensor *>(tensor)));
      } else {
        tensor_refs[tensor_name]->refs++;
      }
    }
  }

  // Merge the refs that reuse the buffer
  for (auto &op : operators) {
    size_t output_size = static_cast<size_t>(op->OutputSize());
    for (size_t i = 0; i < output_size; ++i) {
      Tensor *out_tensor = op->Output(i);
      auto out_tensor_name = out_tensor->name();
      int reuse_input_idx = op->ReuseTensorMapId(i);
      if (reuse_input_idx >= 0) {
        const Tensor *reuse_in_tensor = op->Input(reuse_input_idx);
        if (reuse_in_tensor->is_weight()) {
          continue;
        }
        if (tensor_refs.count(out_tensor_name) == 0) {
          VLOG(2) << "tensor " << out_tensor_name << " is model's output";
          continue;
        }

        // Merge the refs
        auto reuse_in_tensor_name = reuse_in_tensor->name();
        tensor_refs[reuse_in_tensor_name]->refs +=
            tensor_refs.at(out_tensor_name)->refs;
        tensor_refs[out_tensor_name] = tensor_refs.at(reuse_in_tensor_name);
      }
    }
  }

  BufferList used_buf_list;
  BufferList free_buf_list;

  // Simulate the execution of net and allocate memory for tensor
  for (auto &op : operators) {
    VLOG(2) << "Operator " << op->debug_def().name() << "<"
            << op->runtime_type() << ", " << op->debug_def().type() << ">";
    size_t output_size = static_cast<size_t>(op->OutputSize());
    for (size_t i = 0; i < output_size; ++i) {
      Tensor *tensor = op->Output(i);
      auto tensor_name = tensor->name();

      if (tensor_refs.count(tensor_name) == 0) {
        tensor_refs.emplace(tensor_name, std::make_shared<TensorRef>(tensor));
        VLOG(2) << "tensor " << tensor_name << " is model's output";
      }

      std::shared_ptr<TensorRef> tensor_ref = tensor_refs.at(tensor_name);
      // The reused tensor does not need to allocate buffer
      auto essential_tensor_name = tensor_ref->tensor->name();
      if (tensor_name == essential_tensor_name) {
        SimulateAllocateBuffer(tensor_refs.at(tensor_name),
                               &used_buf_list, &free_buf_list);
      } else {
        VLOG(2) << "tensor " << tensor_name << " reuse the "
                << essential_tensor_name;
      }

      auto data_format = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          op->debug_def(), "data_format", static_cast<int>(DataFormat::NONE));
      tensor->set_data_format(static_cast<DataFormat>(data_format));
    }

    size_t input_size = static_cast<size_t>(op->InputSize());
    for (size_t i = 0; i < input_size; ++i) {
      const Tensor *tensor = op->Input(i);
      auto tensor_name = tensor->name();
      if (tensor->is_weight()) {
        continue;
      }
      MACE_CHECK(tensor_refs.count(tensor_name) > 0);
      int ref_num = tensor_refs.at(tensor_name)->refs;
      MACE_CHECK(ref_num > 0);
      tensor_refs[tensor_name]->refs = ref_num - 1;
      if (tensor_refs[tensor_name]->buffer == nullptr) {
        VLOG(3) << "find a model input: " << tensor_name;
        continue;
      }
      if (ref_num == 1) {
        SimulateDeleteBuffer(tensor_refs[tensor_name],
                             &used_buf_list, &free_buf_list);
      }
    }
  }

  ReallyAllocateBuffer(tensor_refs);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace
