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

#include <memory>
#include <string>

#include "mace/ops/common/utils.h"

#include "mace/core/tensor.h"
#include "mace/ops/common/transpose.h"
#include "mace/utils/memory.h"
#include "mace/utils/thread_pool.h"

namespace mace {
namespace ops {
namespace common {
namespace utils {

namespace {
// TODO(luxuhui): Maybe need to adjust
constexpr index_t kCopyBlockSize = 1024 * 1024 * 100;  // 100MB
}

void GetSizeParamFromTensor(const Tensor *size_tensor, index_t *out_height,
                            index_t *out_width) {
  Tensor::MappingGuard size_guard(size_tensor);
  const int *size = size_tensor->data<int>();
  *out_height = size[0];
  *out_width = size[1];
}

template<typename SrcT, typename DstT>
void CopyDataBetweenDiffType(mace::utils::ThreadPool *thread_pool,
                             const SrcT *src, DstT *dst, index_t tensor_size) {
  if (tensor_size < kCopyBlockSize || thread_pool == nullptr) {
    for (index_t i = 0; i < tensor_size; ++i) {
      dst[i] = src[i];
    }
  } else {
    thread_pool->Compute1D(
        [=](index_t start, index_t end, index_t step) {
          for (index_t i = start; i < end; i += step) {
            dst[i] = src[i];
          }
        },
        0, tensor_size, 1);
  }
}

#ifdef MACE_ENABLE_BFLOAT16
template<>
void CopyDataBetweenDiffType<BFloat16, half>(
    mace::utils::ThreadPool *thread_pool,
    const BFloat16 *src, half *dst, index_t tensor_size) {
  if (tensor_size < kCopyBlockSize || thread_pool == nullptr) {
    for (index_t i = 0; i < tensor_size; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  } else {
    thread_pool->Compute1D(
        [=](index_t start, index_t end, index_t step) {
          for (index_t i = start; i < end; i += step) {
            dst[i] = static_cast<float>(src[i]);
          }
        },
        0, tensor_size, 1);
  }
}
#endif

template<typename DstT>
void CopyDataBetweenDiffType(mace::utils::ThreadPool *thread_pool,
                             const void *src, DataType src_type, DstT *dst,
                             index_t tensor_size) {
  MACE_RUN_WITH_TRANPOSE_ENUM(
      src_type,
      CopyDataBetweenDiffType(thread_pool, static_cast<const T *>(src),
                              dst, tensor_size));
}

void CopyDataBetweenType(mace::utils::ThreadPool *thread_pool, const void *src,
                         DataType src_type, void *dst, DataType dst_type,
                         index_t tensor_size) {
  if (src_type == DT_FLOAT16) {
    src_type = DT_HALF;
  }
  if (dst_type == DT_FLOAT16) {
    dst_type = DT_HALF;
  }
  if (src_type == dst_type) {
    auto size = tensor_size * GetEnumTypeSize(src_type);
    if (tensor_size < kCopyBlockSize || thread_pool == nullptr) {
      memcpy(dst, src, size);
    } else {
      thread_pool->Compute1D(
          [=](index_t start, index_t end, index_t step) {
            MACE_UNUSED(step);
            memcpy(static_cast<uint8_t *>(dst) + start,
                   static_cast<const uint8_t *>(src) + start, end - start + 1);
          },
          0, size, kCopyBlockSize);
    }
  } else {
    MACE_RUN_WITH_TRANPOSE_ENUM(
        dst_type,
        CopyDataBetweenDiffType(thread_pool, src, src_type,
                                static_cast<T *>(dst), tensor_size));
  }
}

MaceStatus Transpose(mace::utils::ThreadPool *thread_pool, const void *input,
                     DataType input_data_type,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int> &dst_dims,
                     void *output, DataType output_data_type) {
  MaceStatus status = MaceStatus::MACE_RUNTIME_ERROR;
  if (input_data_type == DT_FLOAT16) {
    input_data_type = DT_HALF;
  }
  if (output_data_type == DT_FLOAT16) {
    output_data_type = DT_HALF;
  }
  MACE_RUN_WITH_TRANPOSE_ENUM(
      output_data_type,
      status = ops::Transpose(thread_pool, input, input_data_type, input_shape,
                              dst_dims, static_cast<T *>(output)));
  return status;
}

MaceStatus DoTransposeConstForCPU(
    OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx) {
  Workspace *ws = context->workspace();

  std::string input_name = op_def->input(input_idx);
  Tensor *input = ws->GetTensor(input_name);

  MemoryType src_mem_type = input->memory_type();
  DataType src_dt = input->dtype();
  MACE_CHECK(src_mem_type == CPU_BUFFER || src_mem_type == GPU_BUFFER,
             "Only support transform CPU_BUFFER or GPU_BUFFER,",
             " but ", input_name, " has ", static_cast<int>(src_mem_type));
  DataType dst_dt = static_cast<DataType>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                *op_def, "T", static_cast<int>(DataType::DT_FLOAT)));
  MACE_CHECK((src_dt == DT_FLOAT || src_dt == DT_HALF) &&
             (dst_dt == DT_FLOAT || dst_dt == DT_HALF),
             "Only support transform DT_FLOAT or DT_HALF,",
             " but ", input_name, " has ", static_cast<int>(src_dt));
  // Only support: half/float -> float, half -> half.
  // Float -> half is not supported.
  if (dst_dt == DT_HALF) {
    MACE_CHECK(src_dt == DT_HALF, "When dst_dt is DT_HALF, "
               "src_dt must be DT_HALF, but ", input_name,
               " has data type of ", static_cast<int>(src_dt));
  }
  MemoryType dst_mem_type = CPU_BUFFER;
  std::vector<index_t> input_shape = input->shape();
  std::vector<index_t> output_shape = input_shape;
  DataFormat op_data_format =
      static_cast<DataFormat>(ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *op_def, "data_format",
          static_cast<int>(DataFormat::NONE)));
  bool cpu_nhwc = (op_data_format == DataFormat::NHWC &&
                   input_shape.size() == 4);
  bool cpu_nchw = (op_data_format == DataFormat::NCHW &&
                   input_shape.size() == 4);
  if (src_mem_type == dst_mem_type &&
      src_dt == dst_dt &&
      (input_shape.size() != 4 || cpu_nhwc)) {
    return MaceStatus::MACE_SUCCESS;
  }
  std::vector<int> dims = {0, 3, 1, 2};
  if (cpu_nchw) {
    for (size_t i = 0; i < dims.size(); ++i) {
      output_shape[i] = input_shape[dims[i]];
    }
  }
  std::string output_name = input_name + std::string("_const_used_by_cpu");
  Tensor *output = ws->GetTensor(output_name);
  MACE_CHECK(output == nullptr || output->shape() == output_shape,
             output_name, " should not exist, ",
             "or it must be of shape ", MakeString(output_shape));
  bool already_transposed = (output != nullptr &&
                             output->shape() == output_shape);
  if (output == nullptr) {
    output = ws->CreateTensor(output_name, context->device()->allocator(),
                              dst_dt, true);
    output->Resize(output_shape);
  }
  op_def->set_input(input_idx, output_name);
  if (already_transposed) {
    return MaceStatus::MACE_SUCCESS;
  }
  Tensor::MappingGuard guard(input);
  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  size_t num_elem = input->size();
  size_t dst_bytes = output->raw_size();

  if (input_shape.size() != 4 || cpu_nhwc) {
    if (src_dt == dst_dt) {
      memcpy(reinterpret_cast<void*>(output_data),
             reinterpret_cast<const void*>(input_data), dst_bytes);
    } else if (src_dt == DT_HALF && dst_dt == DT_FLOAT) {  // half->float
      // Can only be cpu/gpu half to cpu float, no matter 4D or non-4D
      const half *half_input = reinterpret_cast<const half*>(input_data);
      for (size_t i = 0; i < num_elem; ++i) {
        output_data[i] = half_float::half_cast<float>(half_input[i]);
      }
    }
    return MaceStatus::MACE_SUCCESS;
  }
  MaceStatus status;
  status = Transpose(&context->device()->cpu_runtime()->thread_pool(),
                     input_data,
                     src_dt,
                     input_shape,
                     dims,
                     output_data,
                     dst_dt);
  if (status != MaceStatus::MACE_SUCCESS) {
    return status;
  }
  input->MarkUnused();
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus TransposeConstForCPU(
    NetDef *net_def,
    Workspace *ws,
    Device *target_device) {
  // Must be same types in transformer.py,
  // and more types may be added in the future.
  std::unordered_set<std::string> equal_types = {"Eltwise", "Concat"};
  int num_ops = net_def->op_size();
  std::unique_ptr<CPUDevice> cpu_device(
        make_unique<CPUDevice>(
            target_device->cpu_runtime()->num_threads(),
            target_device->cpu_runtime()->policy(),
            &target_device->cpu_runtime()->thread_pool()));

  OpConstructContext construct_context(ws);
  construct_context.set_device(cpu_device.get());
  for (int idx = 0; idx < num_ops; ++idx) {
    OperatorDef *op_def = net_def->mutable_op(idx);
    int end_idx = 1;
    if (equal_types.find(op_def->type()) != equal_types.end()) {
      end_idx =  op_def->input_size();
    }
    DeviceType op_device_type = static_cast<DeviceType>(op_def->device_type());
    if (DeviceType::CPU != op_device_type) {
      continue;
    }
    for (int input_idx = 0; input_idx < end_idx; ++input_idx) {
      Tensor *tensor = ws->GetTensor(op_def->input(input_idx));
      if (!(tensor != nullptr && tensor->is_weight())) {
        continue;
      }
      MaceStatus status = DoTransposeConstForCPU(&construct_context, op_def,
                                                 input_idx);
      if (status != MaceStatus::MACE_SUCCESS) {
        return status;
      }
    }
  }
  return MaceStatus::MACE_SUCCESS;
}
}  // namespace utils
}  // namespace common
}  // namespace ops
}  // namespace mace
