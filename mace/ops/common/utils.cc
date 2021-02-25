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

#include "mace/ops/common/utils.h"

#include "mace/core/tensor.h"
#include "mace/ops/common/transpose.h"
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

}  // namespace utils
}  // namespace common
}  // namespace ops
}  // namespace mace
