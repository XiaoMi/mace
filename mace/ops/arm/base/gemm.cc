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

#include "mace/ops/arm/base/gemm.h"

#include <algorithm>
#include <utility>

#include "mace/ops/arm/base/common_neon.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
void Gemm<T>::Pack4x4(const MatrixMap<const T> &matrix,
                      MatrixMajor dst_major, T *packed_matrix) {
  const index_t rows = matrix.rows();
  const index_t cols = matrix.cols();

  // use the same terminology as GemmLowp:
  // depth is depth, width is the opposite dim other than depth
  // lhs
  index_t width = rows;
  index_t depth = cols;
  index_t width_stride = matrix.rows_stride();
  index_t depth_stride = matrix.cols_stride();
  if (dst_major == RowMajor) {
    // rhs
    std::swap(width, depth);
    std::swap(width_stride, depth_stride);
  }
  const T *data = matrix.data();
  T *packed_ptr = packed_matrix;

  const index_t block_size = 4;
  const index_t depth_padded = RoundUp(depth, static_cast<index_t>(4));

  if (depth_padded > depth) {
    memset(packed_ptr + depth * block_size,
           0,
           sizeof(T) * (depth_padded - depth) * block_size);
  }

  if (dst_major == matrix.matrix_major()) {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        memcpy(packed_ptr, data, sizeof(T) * width);
        memset(packed_ptr + width, 0, sizeof(T) * width_remain);
        data += depth_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t d = 0; d < depth; ++d) {
        float32x4_t vi = vld1q(data);
        vst1q(packed_ptr, vi);
        data += depth_stride;
        packed_ptr += block_size;
      }
    }
  } else {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        for (index_t w = 0; w < width; ++w) {
          packed_ptr[w] = data[w * width_stride + d];
        }  // w
        memset(packed_ptr + width, 0, sizeof(T) * width_remain);
        packed_ptr += block_size;
      }  // d
    } else {
      const T *data0 = data;
      const T *data1 = data + width_stride;
      const T *data2 = data1 + width_stride;
      const T *data3 = data2 + width_stride;

      const index_t depth_block = depth / 4;
      const index_t depth_remain = depth - depth_block * 4;
      for (index_t depth_block_idx = 0; depth_block_idx < depth_block;
           ++depth_block_idx) {
        float32x4_t v0 = vld1q(data0);
        float32x4_t v1 = vld1q(data1);
        float32x4_t v2 = vld1q(data2);
        float32x4_t v3 = vld1q(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        vst1q(packed_ptr, v0123_intertwined.val[0]);
        packed_ptr += 4;

        vst1q(packed_ptr, v0123_intertwined.val[1]);
        packed_ptr += 4;

        vst1q(packed_ptr, v0123n_intertwined.val[0]);
        packed_ptr += 4;

        vst1q(packed_ptr, v0123n_intertwined.val[1]);
        packed_ptr += 4;

        data0 += 4;
        data1 += 4;
        data2 += 4;
        data3 += 4;
      }
      for (index_t d = 0; d < depth_remain; ++d) {
        float32x4_t vi = {*data0, *data1, *data2, *data3};
        vst1q(packed_ptr, vi);
        packed_ptr += 4;

        ++data0;
        ++data1;
        ++data2;
        ++data3;
      }  // d
    }
  }
}

template<typename T>
void Gemm<T>::Pack8x4(const MatrixMap<const T> &matrix,
                      MatrixMajor dst_major, T *packed_matrix) {
  const index_t rows = matrix.rows();
  const index_t cols = matrix.cols();

  // use the same terminology as GemmLowp:
  // depth is depth, width is the opposite dim other than depth
  // lhs
  index_t width = rows;
  index_t depth = cols;
  index_t width_stride = matrix.rows_stride();
  index_t depth_stride = matrix.cols_stride();
  if (dst_major == RowMajor) {
    // rhs
    std::swap(width, depth);
    std::swap(width_stride, depth_stride);
  }
  const T *data = matrix.data();
  T *packed_ptr = packed_matrix;

  const index_t block_size = 8;
  const index_t depth_padded = RoundUp(depth, static_cast<index_t>(4));

  if (depth_padded > depth) {
    memset(packed_ptr + depth * block_size,
           0,
           sizeof(T) * (depth_padded - depth) * block_size);
  }

  if (dst_major == matrix.matrix_major()) {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        memcpy(packed_ptr, data, sizeof(T) * width);
        memset(packed_ptr + width, 0, sizeof(T) * width_remain);
        data += depth_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t d = 0; d < depth; ++d) {
        float32x4_t vi = vld1q(data);
        vst1q(packed_ptr, vi);
        float32x4_t vin = vld1q(data + 4);
        vst1q(packed_ptr + 4, vin);
        data += depth_stride;
        packed_ptr += block_size;
      }
    }
  } else {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        for (index_t w = 0; w < width; ++w) {
          packed_ptr[w] = data[w * width_stride + d];
        }  // w
        memset(packed_ptr + width, 0, sizeof(T) * width_remain);
        packed_ptr += block_size;
      }  // d
    } else {
      const T *data0 = data;
      const T *data1 = data + width_stride;
      const T *data2 = data1 + width_stride;
      const T *data3 = data2 + width_stride;
      const T *data4 = data3 + width_stride;
      const T *data5 = data4 + width_stride;
      const T *data6 = data5 + width_stride;
      const T *data7 = data6 + width_stride;

      const index_t depth_block = depth / 4;
      const index_t depth_remain = depth - depth_block * 4;
      for (index_t depth_block_idx = 0; depth_block_idx < depth_block;
           ++depth_block_idx) {
        float32x4_t v0 = vld1q(data0);
        float32x4_t v1 = vld1q(data1);
        float32x4_t v2 = vld1q(data2);
        float32x4_t v3 = vld1q(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        float32x4_t v4 = vld1q(data4);
        float32x4_t v5 = vld1q(data5);
        float32x4_t v6 = vld1q(data6);
        float32x4_t v7 = vld1q(data7);
        float32x4x2_t v46_intertwined = vzipq_f32(v4, v6);
        float32x4x2_t v57_intertwined = vzipq_f32(v5, v7);
        float32x4x2_t v4567_intertwined =
            vzipq_f32(v46_intertwined.val[0], v57_intertwined.val[0]);
        float32x4x2_t v4567n_intertwined =
            vzipq_f32(v46_intertwined.val[1], v57_intertwined.val[1]);

        vst1q(packed_ptr, v0123_intertwined.val[0]);
        packed_ptr += 4;

        vst1q(packed_ptr, v4567_intertwined.val[0]);
        packed_ptr += 4;

        vst1q(packed_ptr, v0123_intertwined.val[1]);
        packed_ptr += 4;

        vst1q(packed_ptr, v4567_intertwined.val[1]);
        packed_ptr += 4;

        vst1q(packed_ptr, v0123n_intertwined.val[0]);
        packed_ptr += 4;

        vst1q(packed_ptr, v4567n_intertwined.val[0]);
        packed_ptr += 4;

        vst1q(packed_ptr, v0123n_intertwined.val[1]);
        packed_ptr += 4;

        vst1q(packed_ptr, v4567n_intertwined.val[1]);
        packed_ptr += 4;

        data0 += 4;
        data1 += 4;
        data2 += 4;
        data3 += 4;
        data4 += 4;
        data5 += 4;
        data6 += 4;
        data7 += 4;
      }
      for (index_t d = 0; d < depth_remain; ++d) {
        float32x4_t vi = {*data0, *data1, *data2, *data3};
        vst1q(packed_ptr, vi);
        packed_ptr += 4;

        float32x4_t vin = {*data4, *data5, *data6, *data7};
        vst1q(packed_ptr, vin);
        packed_ptr += 4;

        ++data0;
        ++data1;
        ++data2;
        ++data3;
        ++data4;
        ++data5;
        ++data6;
        ++data7;
      }  // d
    }
  }
}

template<typename T>
void Gemm<T>::Unpack4x8(const T *packed_output, MatrixMap<T> *output) {
  const index_t rows = output->rows();
  const index_t cols = output->cols();
  index_t row_stride = output->rows_stride();
  index_t col_stride = output->cols_stride();

  T *output_ptr = output->data();
  const T *packed_ptr = packed_output;

  const index_t block_size = 8;

  // packed_output always has row-major
  if (output->matrix_major() == RowMajor) {
    if (cols < block_size) {
      for (index_t r = 0; r < rows; ++r) {
        memcpy(output_ptr, packed_ptr, sizeof(T) * cols);
        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t r = 0; r < rows; ++r) {
        float32x4_t vi = vld1q(packed_ptr);
        vst1q(output_ptr, vi);
        float32x4_t vin = vld1q(packed_ptr + 4);
        vst1q(output_ptr + 4, vin);

        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    }
  } else {
    // ColMajor
    if (rows < block_size) {
      for (index_t c = 0; c < cols; ++c) {
        for (index_t r = 0; r < rows; ++r) {
          output_ptr[c * col_stride + r] = packed_ptr[r * block_size + c];
        }  // r
      }  // c
    } else {
      const T *data0 = packed_ptr;
      const T *data1 = data0 + block_size;
      const T *data2 = data1 + block_size;
      const T *data3 = data2 + block_size;

      index_t col_block = cols / 4;
      index_t col_remain = cols - col_block * 4;
      for (index_t col_block_idx = 0; col_block_idx < col_block;
           ++col_block_idx) {
        float32x4_t v0 = vld1q(data0);
        float32x4_t v1 = vld1q(data1);
        float32x4_t v2 = vld1q(data2);
        float32x4_t v3 = vld1q(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        vst1q(output_ptr, v0123_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q(output_ptr, v0123_intertwined.val[1]);
        output_ptr += col_stride;

        vst1q(output_ptr, v0123n_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q(output_ptr, v0123n_intertwined.val[1]);
        output_ptr += col_stride;

        data0 += 4;
        data1 += 4;
        data2 += 4;
        data3 += 4;
      }
      for (index_t c = 0; c < col_remain; ++c) {
        float32x4_t vi = {*data0, *data1, *data2, *data3};
        vst1q(output_ptr, vi);
        output_ptr += col_stride;

        ++data0;
        ++data1;
        ++data2;
        ++data3;
      }  // d
    }
  }
}

template<typename T>
void Gemm<T>::Unpack8x8(const T *packed_output, MatrixMap<T> *output) {
  const index_t rows = output->rows();
  const index_t cols = output->cols();
  index_t row_stride = output->rows_stride();
  index_t col_stride = output->cols_stride();

  T *output_ptr = output->data();
  const T *packed_ptr = packed_output;

  const index_t block_size = 8;

  // packed_output always has row-major
  if (output->matrix_major() == RowMajor) {
    if (cols < block_size) {
      for (index_t r = 0; r < rows; ++r) {
        memcpy(output_ptr, packed_ptr, sizeof(T) * cols);
        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t r = 0; r < rows; ++r) {
        float32x4_t vi = vld1q(packed_ptr);
        vst1q(output_ptr, vi);
        float32x4_t vin = vld1q(packed_ptr + 4);
        vst1q(output_ptr + 4, vin);

        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    }
  } else {
    // ColMajor
    if (rows < block_size) {
      for (index_t c = 0; c < cols; ++c) {
        for (index_t r = 0; r < rows; ++r) {
          output_ptr[c * col_stride + r] = packed_ptr[r * block_size + c];
        }  // r
      }  // c
    } else {
      const T *data0 = packed_ptr;
      const T *data1 = data0 + block_size;
      const T *data2 = data1 + block_size;
      const T *data3 = data2 + block_size;
      const T *data4 = data3 + block_size;
      const T *data5 = data4 + block_size;
      const T *data6 = data5 + block_size;
      const T *data7 = data6 + block_size;

      index_t col_block = cols / 4;
      index_t col_remain = cols - col_block * 4;
      for (index_t col_block_idx = 0; col_block_idx < col_block;
           ++col_block_idx) {
        float32x4_t v0 = vld1q(data0);
        float32x4_t v1 = vld1q(data1);
        float32x4_t v2 = vld1q(data2);
        float32x4_t v3 = vld1q(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        float32x4_t v4 = vld1q(data4);
        float32x4_t v5 = vld1q(data5);
        float32x4_t v6 = vld1q(data6);
        float32x4_t v7 = vld1q(data7);
        float32x4x2_t v46_intertwined = vzipq_f32(v4, v6);
        float32x4x2_t v57_intertwined = vzipq_f32(v5, v7);
        float32x4x2_t v4567_intertwined =
            vzipq_f32(v46_intertwined.val[0], v57_intertwined.val[0]);
        float32x4x2_t v4567n_intertwined =
            vzipq_f32(v46_intertwined.val[1], v57_intertwined.val[1]);

        vst1q(output_ptr, v0123_intertwined.val[0]);
        vst1q(output_ptr + 4, v4567_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q(output_ptr, v0123_intertwined.val[1]);
        vst1q(output_ptr + 4, v4567_intertwined.val[1]);
        output_ptr += col_stride;

        vst1q(output_ptr, v0123n_intertwined.val[0]);
        vst1q(output_ptr + 4, v4567n_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q(output_ptr, v0123n_intertwined.val[1]);
        vst1q(output_ptr + 4, v4567n_intertwined.val[1]);
        output_ptr += col_stride;

        data0 += 4;
        data1 += 4;
        data2 += 4;
        data3 += 4;
        data4 += 4;
        data5 += 4;
        data6 += 4;
        data7 += 4;
      }
      for (index_t c = 0; c < col_remain; ++c) {
        float32x4_t vi = {*data0, *data1, *data2, *data3};
        vst1q(output_ptr, vi);
        float32x4_t vin = {*data4, *data5, *data6, *data7};
        vst1q(output_ptr + 4, vin);
        output_ptr += col_stride;

        ++data0;
        ++data1;
        ++data2;
        ++data3;
        ++data4;
        ++data5;
        ++data6;
        ++data7;
      }  // d
    }
  }
}

template<typename T>
void Gemm<T>::PackLhs(const MatrixMap<const T> &lhs, T *packed_lhs) {
#ifdef __aarch64__
  Pack8x4(lhs, ColMajor, packed_lhs);
#else
  Pack4x4(lhs, ColMajor, packed_lhs);
#endif
}

template<typename T>
void Gemm<T>::PackRhs(const MatrixMap<const T> &rhs, T *packed_rhs) {
  Pack8x4(rhs, RowMajor, packed_rhs);
}

template<typename T>
void Gemm<T>::UnpackOutput(const T *packed_output, MatrixMap<T> *output) {
#ifdef __aarch64__
  Unpack8x8(packed_output, output);
#else
  Unpack4x8(packed_output, output);
#endif
}

template<typename T>
MaceStatus Gemm<T>::Compute(
    const OpContext *context, const Tensor *lhs, const Tensor *rhs,
    const index_t batch, const index_t rows, const index_t cols,
    const index_t depth, const MatrixMajor lhs_major,
    const MatrixMajor rhs_major, const MatrixMajor output_major,
    const bool lhs_batched, const bool rhs_batched, Tensor *output) {
  MACE_CHECK(output->size() == batch * rows * cols,
             "Need resize output tensor before call gemm.");
  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard output_guard(output);
  const T *lhs_data = lhs->data<T>();
  const T *rhs_data = rhs->data<T>();
  T *output_data = output->mutable_data<T>();

#ifdef __aarch64__
  const index_t row_block_size = 8;
#else
  const index_t row_block_size = 4;
#endif
  const index_t col_block_size = 8;
  const index_t depth_block_size = 4;
  const index_t row_block_count = RoundUpDiv(rows, row_block_size);
  const index_t col_block_count = RoundUpDiv(cols, col_block_size);
  const index_t rows_padded = RoundUp(rows, row_block_size);
  const index_t cols_padded = RoundUp(cols, col_block_size);
  const index_t depth_padded = RoundUp(depth, depth_block_size);

  ScratchBuffer *scratch = context->device()->scratch_buffer();

  index_t packed_lhs_size =
      PadAlignSize(sizeof(T) * rows_padded * depth_padded);
  index_t packed_rhs_size =
      PadAlignSize(sizeof(T) * depth_padded * cols_padded);
  index_t packed_output_size =
      PadAlignSize(sizeof(T) * rows_padded * cols_padded);
  // resize to the total size of lhs & rhs & output anyway,
  // in case we do not cache const tensor for saving memory
  MACE_RETURN_IF_ERROR(scratch->GrowSize(
      packed_lhs_size + packed_rhs_size + packed_output_size));
  T *packed_lhs_data =
      scratch->Scratch(packed_lhs_size).mutable_data<T>();
  T *packed_rhs_data =
      scratch->Scratch(packed_rhs_size).mutable_data<T>();
  T *packed_output_data =
      scratch->Scratch(packed_output_size).mutable_data<T>();

  int cache_side = kNoCache;
  if (cached_ == kCacheLhs) {
    packed_lhs_data = pack_cache_.mutable_data<T>();
  } else if (cached_ == kCacheRhs) {
    packed_rhs_data = pack_cache_.mutable_data<T>();
  } else if (should_cache_pack_) {
    if (lhs->is_weight() && (!lhs_batched || batch == 1)) {
      cache_side = kCacheLhs;
      pack_cache_.Resize(packed_lhs_size);
      packed_lhs_data = pack_cache_.mutable_data<T>();
    } else if (rhs->is_weight() && (!rhs_batched || batch == 1)) {
      cache_side = kCacheRhs;
      pack_cache_.Resize(packed_rhs_size);
      packed_rhs_data = pack_cache_.mutable_data<T>();
    }
  }

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  for (index_t b = 0; b < batch; ++b) {
    MatrixMap<const T>
        lhs_matrix
        (lhs_data + static_cast<index_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const T>
        rhs_matrix
        (rhs_data + static_cast<index_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<T> output_matrix
        (output_data + b * rows * cols, output_major, rows, cols);

    // pack lhs
    if (cached_ != kCacheLhs) {
      thread_pool.Compute1D([=, &lhs_matrix](index_t start,
                                             index_t end,
                                             index_t step) {
        for (index_t row_block_idx = start; row_block_idx < end;
             row_block_idx += step) {
          const index_t start_row = row_block_idx * row_block_size;
          const index_t
              row_block_len = std::min(row_block_size, rows - start_row);
          T *packed_lhs_data_block =
              packed_lhs_data + row_block_idx * row_block_size * depth_padded;
          PackLhs(lhs_matrix.block(start_row, 0, row_block_len, depth),
                  packed_lhs_data_block);
        }
      }, 0, row_block_count, 1);

      if (cache_side == kCacheLhs) {
        cached_ = kCacheLhs;
        if (lhs->UnderlyingBuffer()->OnHost()) {
          AdviseFree(reinterpret_cast<void *>(const_cast<T *>(lhs->data<
                         T>())),
                     lhs->raw_size());
        }
      }
    }

    // pack rhs
    if (cached_ != kCacheRhs) {
      thread_pool.Compute1D([=, &rhs_matrix](index_t start,
                                             index_t end,
                                             index_t step) {
        for (index_t col_block_idx = start; col_block_idx < end;
             col_block_idx += step) {
          const index_t start_col = col_block_idx * col_block_size;
          const index_t
              col_block_len = std::min(col_block_size, cols - start_col);
          T *packed_rhs_data_block =
              packed_rhs_data + col_block_idx * col_block_size * depth_padded;
          PackRhs(rhs_matrix.block(0, start_col, depth, col_block_len),
                  packed_rhs_data_block);
        }
      }, 0, col_block_count, 1);

      if (cache_side == kCacheRhs) {
        cached_ = kCacheRhs;
        if (rhs->UnderlyingBuffer()->OnHost()) {
          AdviseFree(reinterpret_cast<void *>(const_cast<T *>(rhs->data<
                         T>())),
                     rhs->raw_size());
        }
      }
    }

    // multiply lhs and rhs
    thread_pool.Compute1D([=, &output_matrix](index_t start,
                                              index_t end,
                                              index_t step) {
      for (index_t row_block_idx = start; row_block_idx < end;
           row_block_idx += step) {
        const index_t start_row = row_block_idx * row_block_size;
        const index_t
            row_block_len = std::min(row_block_size, rows - start_row);
        const T *packed_lhs_data_block =
            packed_lhs_data + row_block_idx * row_block_size * depth_padded;

        for (index_t col_block_idx = 0; col_block_idx < col_block_count;
             ++col_block_idx) {
          const index_t start_col = col_block_idx * col_block_size;
          const index_t
              col_block_len = std::min(col_block_size, cols - start_col);
          const T *packed_rhs_data_block =
              packed_rhs_data + col_block_idx * col_block_size * depth_padded;
          T *packed_output_data_block =
              packed_output_data + row_block_idx * row_block_size * cols_padded
                  + col_block_idx * col_block_size;
          ComputeBlock(packed_lhs_data_block,
                       packed_rhs_data_block,
                       depth_padded,
                       packed_output_data_block);
          MatrixMap<T> output_block = output_matrix.block(start_row,
                                                          start_col,
                                                          row_block_len,
                                                          col_block_len);
          UnpackOutput(packed_output_data_block, &output_block);
        }  // col_block_idx
      }  // row_block_idx
    }, 0, row_block_count, 1);
  }  // b

  return MaceStatus::MACE_SUCCESS;
}

void RegisterGemmDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Gemm<float>, delegator::GemmParam,
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, float, ImplType::NEON));

  MACE_REGISTER_BF16_DELEGATOR(
      registry, Gemm<BFloat16>, delegator::GemmParam,
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, BFloat16, ImplType::NEON));
}
}  // namespace arm
}  // namespace ops
}  // namespace mace
