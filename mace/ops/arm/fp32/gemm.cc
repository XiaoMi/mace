// Copyright 2019 The MACE Authors. All Rights Reserved.
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


#include "mace/ops/arm/fp32/gemm.h"

#include <arm_neon.h>
#include <algorithm>
#include <utility>

#include "mace/port/env.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

enum { kNoCache, kCacheLhs, kCacheRhs };

MaceStatus Gemm::Compute(const OpContext *context,
                         const Tensor *lhs,
                         const Tensor *rhs,
                         const index_t batch,
                         const index_t rows,
                         const index_t cols,
                         const index_t depth,
                         const MatrixMajor lhs_major,
                         const MatrixMajor rhs_major,
                         const MatrixMajor output_major,
                         const bool lhs_batched,
                         const bool rhs_batched,
                         Tensor *output) {
  MACE_CHECK(output->size() == batch * rows * cols,
             "Need resize output tensor before call gemm.");
  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard output_guard(output);
  const float *lhs_data = lhs->data<float>();
  const float *rhs_data = rhs->data<float>();
  float *output_data = output->mutable_data<float>();

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
      PadAlignSize(sizeof(float) * rows_padded * depth_padded);
  index_t packed_rhs_size =
      PadAlignSize(sizeof(float) * depth_padded * cols_padded);
  index_t packed_output_size =
      PadAlignSize(sizeof(float) * rows_padded * cols_padded);
  // resize to the total size of lhs & rhs & output anyway,
  // in case we do not cache const tensor for saving memory
  MACE_RETURN_IF_ERROR(scratch->GrowSize(
      packed_lhs_size + packed_rhs_size + packed_output_size));
  float *packed_lhs_data =
      scratch->Scratch(packed_lhs_size).mutable_data<float>();
  float *packed_rhs_data =
      scratch->Scratch(packed_rhs_size).mutable_data<float>();
  float *packed_output_data =
      scratch->Scratch(packed_output_size).mutable_data<float>();

  int cache_side = kNoCache;
  if (cached_ == kCacheLhs) {
    packed_lhs_data = pack_cache_.mutable_data<float>();
  } else if (cached_ == kCacheRhs) {
    packed_rhs_data = pack_cache_.mutable_data<float>();
  } else if (should_cache_pack_) {
    if (lhs->is_weight() && (!lhs_batched || batch == 1)) {
      cache_side = kCacheLhs;
      pack_cache_.Resize(packed_lhs_size);
      packed_lhs_data = pack_cache_.mutable_data<float>();
    } else if (rhs->is_weight() && (!rhs_batched || batch == 1)) {
      cache_side = kCacheRhs;
      pack_cache_.Resize(packed_rhs_size);
      packed_rhs_data = pack_cache_.mutable_data<float>();
    }
  }

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  for (index_t b = 0; b < batch; ++b) {
    MatrixMap<const float>
        lhs_matrix
        (lhs_data + static_cast<index_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const float>
        rhs_matrix
        (rhs_data + static_cast<index_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<float> output_matrix
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
          float *packed_lhs_data_block =
              packed_lhs_data + row_block_idx * row_block_size * depth_padded;
          PackLhs(lhs_matrix.block(start_row, 0, row_block_len, depth),
                  packed_lhs_data_block);
        }
      }, 0, row_block_count, 1);

      if (cache_side == kCacheLhs) {
        cached_ = kCacheLhs;
        if (lhs->UnderlyingBuffer()->OnHost()) {
          AdviseFree(reinterpret_cast<void *>(const_cast<float *>(lhs->data<
                         float>())),
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
          float *packed_rhs_data_block =
              packed_rhs_data + col_block_idx * col_block_size * depth_padded;
          PackRhs(rhs_matrix.block(0, start_col, depth, col_block_len),
                  packed_rhs_data_block);
        }
      }, 0, col_block_count, 1);

      if (cache_side == kCacheRhs) {
        cached_ = kCacheRhs;
        if (rhs->UnderlyingBuffer()->OnHost()) {
          AdviseFree(reinterpret_cast<void *>(const_cast<float *>(rhs->data<
                         float>())),
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
        const float *packed_lhs_data_block =
            packed_lhs_data + row_block_idx * row_block_size * depth_padded;

        for (index_t col_block_idx = 0; col_block_idx < col_block_count;
             ++col_block_idx) {
          const index_t start_col = col_block_idx * col_block_size;
          const index_t
              col_block_len = std::min(col_block_size, cols - start_col);
          const float *packed_rhs_data_block =
              packed_rhs_data + col_block_idx * col_block_size * depth_padded;
          float *packed_output_data_block =
              packed_output_data + row_block_idx * row_block_size * cols_padded
                  + col_block_idx * col_block_size;
          ComputeBlock(packed_lhs_data_block,
                       packed_rhs_data_block,
                       depth_padded,
                       packed_output_data_block);
          MatrixMap<float> output_block = output_matrix.block(start_row,
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

void Gemm::ComputeBlock(const float *packed_lhs_data,
                        const float *packed_rhs_data,
                        const index_t depth_padded,
                        float *packed_output_data) {
  /* Ref:
  for (index_t r = 0; r < block_size; ++r) {
    for (index_t c = 0; c < block_size; ++c) {
      float sum = 0;
      for (index_t d = 0; d < depth; ++d) {
        // (r, d) * (d, c)
        sum += packed_lhs_data[d * r_block_size + r]
            * packed_rhs_data[d * c_block_size + c];
      }
      packed_output_data[r * c_block_size + c] = sum;
    }
  }
  */
  const float *lhs_ptr = packed_lhs_data;
  const float *rhs_ptr = packed_rhs_data;

  const index_t depth_block_count = depth_padded / 4;

#ifdef __aarch64__
  // Register layout: (8x4) x (4,8)
  //
  //                               +--------+--------+
  //                               | v8 ... | v9 ... |
  //                       Rhs     +--------+--------+
  //                               | v10... | v11... |
  //                               +--------+--------+
  //                               | v12... | v13... |
  //                               +--------+--------+
  //                               | v14... | v15... |
  //                               +--------+--------+
  //
  //          Lhs
  //
  //  +----+----+----+----+  -  -  +--------+--------+
  //  | v0 | v2 | v4 | v6 |        | v16... | v17... |
  //  | .  |    |    |    |        | v18... | v19... |
  //  | .  |    |    |    |        | v20... | v21... |
  //  | .  |    |    |    |        | v22... | v23... |
  //  +----+----|----+----+        +--------+--------+
  //  | v1 | v3 | v5 | v7 |        | v24... | v25... |
  //  | .  |    |    |    |        | v26... | v27... |
  //  | .  |    |    |    |        | v28... | v29... |
  //  | .  |    |    |    |        | v30... | v31... |
  //  +----+----|----+----+        +--------+--------+
  //
  //                                    Accumulator
  //

  if (depth_block_count > 0) {
    index_t r_depth_block_count = depth_block_count;
    // just make compiler happy
    MACE_UNUSED(r_depth_block_count);

    asm volatile(
        "dup v16.4s, wzr \n"
        "dup v17.4s, wzr \n"
        "dup v18.4s, wzr \n"
        "dup v19.4s, wzr \n"
        "dup v20.4s, wzr \n"
        "dup v21.4s, wzr \n"
        "dup v22.4s, wzr \n"
        "dup v23.4s, wzr \n"
        "dup v24.4s, wzr \n"
        "dup v25.4s, wzr \n"
        "dup v26.4s, wzr \n"
        "dup v27.4s, wzr \n"
        "dup v28.4s, wzr \n"
        "dup v29.4s, wzr \n"
        "dup v30.4s, wzr \n"
        "dup v31.4s, wzr \n"

        // prelogue
        "ld1 {v0.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v2.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v3.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v4.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v5.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v6.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v7.4s}, [%[lhs_ptr]], #16 \n"

        "ld1 {v8.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v9.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v10.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v11.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v12.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v13.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v14.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v15.4s}, [%[rhs_ptr]], #16 \n"

        "subs %[r_depth_block_count], %[r_depth_block_count], #1 \n"
        "beq 1f\n"

        "0: \n"
        "fmla v16.4s, v8.4s, v0.s[0] \n"
        "fmla v17.4s, v9.4s, v0.s[0] \n"
        "fmla v18.4s, v8.4s, v0.s[1] \n"
        "fmla v19.4s, v9.4s, v0.s[1] \n"
        "fmla v20.4s, v8.4s, v0.s[2] \n"
        "fmla v21.4s, v9.4s, v0.s[2] \n"
        "fmla v22.4s, v8.4s, v0.s[3] \n"
        "fmla v23.4s, v9.4s, v0.s[3] \n"

        "ld1 {v0.4s}, [%[lhs_ptr]], #16 \n"

        "fmla v24.4s, v8.4s, v1.s[0] \n"
        "fmla v25.4s, v9.4s, v1.s[0] \n"
        "fmla v26.4s, v8.4s, v1.s[1] \n"
        "fmla v27.4s, v9.4s, v1.s[1] \n"
        "fmla v28.4s, v8.4s, v1.s[2] \n"
        "fmla v29.4s, v9.4s, v1.s[2] \n"
        "fmla v30.4s, v8.4s, v1.s[3] \n"
        "fmla v31.4s, v9.4s, v1.s[3] \n"

        "ld1 {v1.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v8.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v9.4s}, [%[rhs_ptr]], #16 \n"

        "fmla v16.4s, v10.4s, v2.s[0] \n"
        "fmla v17.4s, v11.4s, v2.s[0] \n"
        "fmla v18.4s, v10.4s, v2.s[1] \n"
        "fmla v19.4s, v11.4s, v2.s[1] \n"
        "fmla v20.4s, v10.4s, v2.s[2] \n"
        "fmla v21.4s, v11.4s, v2.s[2] \n"
        "fmla v22.4s, v10.4s, v2.s[3] \n"
        "fmla v23.4s, v11.4s, v2.s[3] \n"

        "ld1 {v2.4s}, [%[lhs_ptr]], #16 \n"

        "fmla v24.4s, v10.4s, v3.s[0] \n"
        "fmla v25.4s, v11.4s, v3.s[0] \n"
        "fmla v26.4s, v10.4s, v3.s[1] \n"
        "fmla v27.4s, v11.4s, v3.s[1] \n"
        "fmla v28.4s, v10.4s, v3.s[2] \n"
        "fmla v29.4s, v11.4s, v3.s[2] \n"
        "fmla v30.4s, v10.4s, v3.s[3] \n"
        "fmla v31.4s, v11.4s, v3.s[3] \n"

        "ld1 {v3.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v10.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v11.4s}, [%[rhs_ptr]], #16 \n"

        "fmla v16.4s, v12.4s, v4.s[0] \n"
        "fmla v17.4s, v13.4s, v4.s[0] \n"
        "fmla v18.4s, v12.4s, v4.s[1] \n"
        "fmla v19.4s, v13.4s, v4.s[1] \n"
        "fmla v20.4s, v12.4s, v4.s[2] \n"
        "fmla v21.4s, v13.4s, v4.s[2] \n"
        "fmla v22.4s, v12.4s, v4.s[3] \n"
        "fmla v23.4s, v13.4s, v4.s[3] \n"

        "ld1 {v4.4s}, [%[lhs_ptr]], #16 \n"

        "fmla v24.4s, v12.4s, v5.s[0] \n"
        "fmla v25.4s, v13.4s, v5.s[0] \n"
        "fmla v26.4s, v12.4s, v5.s[1] \n"
        "fmla v27.4s, v13.4s, v5.s[1] \n"
        "fmla v28.4s, v12.4s, v5.s[2] \n"
        "fmla v29.4s, v13.4s, v5.s[2] \n"
        "fmla v30.4s, v12.4s, v5.s[3] \n"
        "fmla v31.4s, v13.4s, v5.s[3] \n"

        "ld1 {v5.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v12.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v13.4s}, [%[rhs_ptr]], #16 \n"

        "fmla v16.4s, v14.4s, v6.s[0] \n"
        "fmla v17.4s, v15.4s, v6.s[0] \n"
        "fmla v18.4s, v14.4s, v6.s[1] \n"
        "fmla v19.4s, v15.4s, v6.s[1] \n"
        "fmla v20.4s, v14.4s, v6.s[2] \n"
        "fmla v21.4s, v15.4s, v6.s[2] \n"
        "fmla v22.4s, v14.4s, v6.s[3] \n"
        "fmla v23.4s, v15.4s, v6.s[3] \n"

        "ld1 {v6.4s}, [%[lhs_ptr]], #16 \n"

        "subs %[r_depth_block_count], %[r_depth_block_count], #1 \n"

        "fmla v24.4s, v14.4s, v7.s[0] \n"
        "fmla v25.4s, v15.4s, v7.s[0] \n"
        "fmla v26.4s, v14.4s, v7.s[1] \n"
        "fmla v27.4s, v15.4s, v7.s[1] \n"
        "fmla v28.4s, v14.4s, v7.s[2] \n"
        "fmla v29.4s, v15.4s, v7.s[2] \n"
        "fmla v30.4s, v14.4s, v7.s[3] \n"
        "fmla v31.4s, v15.4s, v7.s[3] \n"

        "ld1 {v7.4s}, [%[lhs_ptr]], #16 \n"
        "ld1 {v14.4s}, [%[rhs_ptr]], #16 \n"
        "ld1 {v15.4s}, [%[rhs_ptr]], #16 \n"

        "bne 0b \n"

        // prologue
        "1:\n"
        "fmla v16.4s, v8.4s, v0.s[0] \n"
        "fmla v17.4s, v9.4s, v0.s[0] \n"
        "fmla v18.4s, v8.4s, v0.s[1] \n"
        "fmla v19.4s, v9.4s, v0.s[1] \n"
        "fmla v20.4s, v8.4s, v0.s[2] \n"
        "fmla v21.4s, v9.4s, v0.s[2] \n"
        "fmla v22.4s, v8.4s, v0.s[3] \n"
        "fmla v23.4s, v9.4s, v0.s[3] \n"

        "fmla v24.4s, v8.4s, v1.s[0] \n"
        "fmla v25.4s, v9.4s, v1.s[0] \n"
        "fmla v26.4s, v8.4s, v1.s[1] \n"
        "fmla v27.4s, v9.4s, v1.s[1] \n"
        "fmla v28.4s, v8.4s, v1.s[2] \n"
        "fmla v29.4s, v9.4s, v1.s[2] \n"
        "fmla v30.4s, v8.4s, v1.s[3] \n"
        "fmla v31.4s, v9.4s, v1.s[3] \n"

        "fmla v16.4s, v10.4s, v2.s[0] \n"
        "fmla v17.4s, v11.4s, v2.s[0] \n"
        "fmla v18.4s, v10.4s, v2.s[1] \n"
        "fmla v19.4s, v11.4s, v2.s[1] \n"
        "fmla v20.4s, v10.4s, v2.s[2] \n"
        "fmla v21.4s, v11.4s, v2.s[2] \n"
        "fmla v22.4s, v10.4s, v2.s[3] \n"
        "fmla v23.4s, v11.4s, v2.s[3] \n"

        "fmla v24.4s, v10.4s, v3.s[0] \n"
        "fmla v25.4s, v11.4s, v3.s[0] \n"
        "fmla v26.4s, v10.4s, v3.s[1] \n"
        "fmla v27.4s, v11.4s, v3.s[1] \n"
        "fmla v28.4s, v10.4s, v3.s[2] \n"
        "fmla v29.4s, v11.4s, v3.s[2] \n"
        "fmla v30.4s, v10.4s, v3.s[3] \n"
        "fmla v31.4s, v11.4s, v3.s[3] \n"

        "fmla v16.4s, v12.4s, v4.s[0] \n"
        "fmla v17.4s, v13.4s, v4.s[0] \n"
        "fmla v18.4s, v12.4s, v4.s[1] \n"
        "fmla v19.4s, v13.4s, v4.s[1] \n"
        "fmla v20.4s, v12.4s, v4.s[2] \n"
        "fmla v21.4s, v13.4s, v4.s[2] \n"
        "fmla v22.4s, v12.4s, v4.s[3] \n"
        "fmla v23.4s, v13.4s, v4.s[3] \n"

        "fmla v24.4s, v12.4s, v5.s[0] \n"
        "fmla v25.4s, v13.4s, v5.s[0] \n"
        "fmla v26.4s, v12.4s, v5.s[1] \n"
        "fmla v27.4s, v13.4s, v5.s[1] \n"
        "fmla v28.4s, v12.4s, v5.s[2] \n"
        "fmla v29.4s, v13.4s, v5.s[2] \n"
        "fmla v30.4s, v12.4s, v5.s[3] \n"
        "fmla v31.4s, v13.4s, v5.s[3] \n"

        "fmla v16.4s, v14.4s, v6.s[0] \n"
        "fmla v17.4s, v15.4s, v6.s[0] \n"
        "fmla v18.4s, v14.4s, v6.s[1] \n"
        "fmla v19.4s, v15.4s, v6.s[1] \n"
        "fmla v20.4s, v14.4s, v6.s[2] \n"
        "fmla v21.4s, v15.4s, v6.s[2] \n"
        "fmla v22.4s, v14.4s, v6.s[3] \n"
        "fmla v23.4s, v15.4s, v6.s[3] \n"

        "fmla v24.4s, v14.4s, v7.s[0] \n"
        "fmla v25.4s, v15.4s, v7.s[0] \n"
        "fmla v26.4s, v14.4s, v7.s[1] \n"
        "fmla v27.4s, v15.4s, v7.s[1] \n"
        "fmla v28.4s, v14.4s, v7.s[2] \n"
        "fmla v29.4s, v15.4s, v7.s[2] \n"
        "fmla v30.4s, v14.4s, v7.s[3] \n"
        "fmla v31.4s, v15.4s, v7.s[3] \n"

        "st1 {v16.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v17.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v18.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v19.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v20.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v21.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v22.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v23.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v24.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v25.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v26.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v27.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v28.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v29.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v30.4s}, [%[packed_output_data]], #16 \n"
        "st1 {v31.4s}, [%[packed_output_data]], #16 \n"
    :  // outputs
    [lhs_ptr] "+r"(lhs_ptr),
    [rhs_ptr] "+r"(rhs_ptr),
    [packed_output_data] "+r"(packed_output_data),
    [r_depth_block_count] "+r"(r_depth_block_count)
    :  // inputs
    :  // clabbers
    "cc", "memory",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
  }
#else  // armeabi-v7a
  // Register layout: (4x4) x (4,8)
  //
  //                               +--------+--------+
  //                               | q4 ... | q5 ... |
  //                       Rhs     +--------+--------+
  //                               | q6 ... | q7 ... |
  //                               +--------+--------+
  //                               | q4 ... | q5 ... |
  //                               +--------+--------+
  //                               | q6 ... | q7 ... |
  //                               +--------+--------+
  //
  //          Lhs
  //
  //  +----+----+----+----+  -  -  +--------+--------+
  //  | q0 | q1 | q2 | q3 |        | q8...  | q9...  |
  //  | .  |    |    |    |        | q10... | q11... |
  //  | .  |    |    |    |        | q12... | q13... |
  //  | .  |    |    |    |        | q14... | q15... |
  //  +----+----+----+----+        +--------+--------+
  //
  //                                    Accumulator
  //

  if (depth_block_count > 0) {
    index_t r_depth_block_count = depth_block_count;
    // just make compiler happy
    MACE_UNUSED(r_depth_block_count);

    asm volatile(
    "mov r0, #0\n"
    "vdup.f32 q8, r0 \n"
    "vdup.f32 q9, r0 \n"
    "vdup.f32 q10, r0 \n"
    "vdup.f32 q11, r0 \n"
    "vdup.f32 q12, r0 \n"
    "vdup.f32 q13, r0 \n"
    "vdup.f32 q14, r0 \n"
    "vdup.f32 q15, r0 \n"

    // prelogue
    "vld1.f32 {d0-d1}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d2-d3}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d4-d5}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d6-d7}, [%[lhs_ptr]]! \n"

    "vld1.f32 {d8-d9}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d10-d11}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d12-d13}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d14-d15}, [%[rhs_ptr]]! \n"

    "subs %[r_depth_block_count], %[r_depth_block_count], #1 \n"
    "beq 1f\n"

    "0: \n"

    "vmla.f32 q8, q4, d0[0] \n"
    "vmla.f32 q9, q5, d0[0] \n"
    "vmla.f32 q10, q4, d0[1] \n"
    "vmla.f32 q11, q5, d0[1] \n"
    "vmla.f32 q12, q4, d1[0] \n"
    "vmla.f32 q13, q5, d1[0] \n"
    "vmla.f32 q14, q4, d1[1] \n"
    "vmla.f32 q15, q5, d1[1] \n"

    "vld1.f32 {d0-d1}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d8-d9}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d10-d11}, [%[rhs_ptr]]! \n"

    "vmla.f32 q8, q6, d2[0] \n"
    "vmla.f32 q9, q7, d2[0] \n"
    "vmla.f32 q10, q6, d2[1] \n"
    "vmla.f32 q11, q7, d2[1] \n"
    "vmla.f32 q12, q6, d3[0] \n"
    "vmla.f32 q13, q7, d3[0] \n"
    "vmla.f32 q14, q6, d3[1] \n"
    "vmla.f32 q15, q7, d3[1] \n"

    "vld1.f32 {d2-d3}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d12-d13}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d14-d15}, [%[rhs_ptr]]! \n"

    "vmla.f32 q8, q4, d4[0] \n"
    "vmla.f32 q9, q5, d4[0] \n"
    "vmla.f32 q10, q4, d4[1] \n"
    "vmla.f32 q11, q5, d4[1] \n"
    "vmla.f32 q12, q4, d5[0] \n"
    "vmla.f32 q13, q5, d5[0] \n"
    "vmla.f32 q14, q4, d5[1] \n"
    "vmla.f32 q15, q5, d5[1] \n"

    "vld1.f32 {d4-d5}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d8-d9}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d10-d11}, [%[rhs_ptr]]! \n"

    "subs %[r_depth_block_count], %[r_depth_block_count], #1 \n"

    "vmla.f32 q8, q6, d6[0] \n"
    "vmla.f32 q9, q7, d6[0] \n"
    "vmla.f32 q10, q6, d6[1] \n"
    "vmla.f32 q11, q7, d6[1] \n"
    "vmla.f32 q12, q6, d7[0] \n"
    "vmla.f32 q13, q7, d7[0] \n"
    "vmla.f32 q14, q6, d7[1] \n"
    "vmla.f32 q15, q7, d7[1] \n"

    "vld1.f32 {d6-d7}, [%[lhs_ptr]]! \n"
    "vld1.f32 {d12-d13}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d14-d15}, [%[rhs_ptr]]! \n"

    "bne 0b \n"

    // prologue
    "1:\n"
    "vmla.f32 q8, q4, d0[0] \n"
    "vmla.f32 q9, q5, d0[0] \n"
    "vmla.f32 q10, q4, d0[1] \n"
    "vmla.f32 q11, q5, d0[1] \n"
    "vmla.f32 q12, q4, d1[0] \n"
    "vmla.f32 q13, q5, d1[0] \n"
    "vmla.f32 q14, q4, d1[1] \n"
    "vmla.f32 q15, q5, d1[1] \n"

    "vld1.f32 {d8-d9}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d10-d11}, [%[rhs_ptr]]! \n"

    "vmla.f32 q8, q6, d2[0] \n"
    "vmla.f32 q9, q7, d2[0] \n"
    "vmla.f32 q10, q6, d2[1] \n"
    "vmla.f32 q11, q7, d2[1] \n"
    "vmla.f32 q12, q6, d3[0] \n"
    "vmla.f32 q13, q7, d3[0] \n"
    "vmla.f32 q14, q6, d3[1] \n"
    "vmla.f32 q15, q7, d3[1] \n"

    "vld1.f32 {d12-d13}, [%[rhs_ptr]]! \n"
    "vld1.f32 {d14-d15}, [%[rhs_ptr]]! \n"

    "vmla.f32 q8, q4, d4[0] \n"
    "vmla.f32 q9, q5, d4[0] \n"
    "vmla.f32 q10, q4, d4[1] \n"
    "vmla.f32 q11, q5, d4[1] \n"
    "vmla.f32 q12, q4, d5[0] \n"
    "vmla.f32 q13, q5, d5[0] \n"
    "vmla.f32 q14, q4, d5[1] \n"
    "vmla.f32 q15, q5, d5[1] \n"

    "vmla.f32 q8, q6, d6[0] \n"
    "vmla.f32 q9, q7, d6[0] \n"
    "vmla.f32 q10, q6, d6[1] \n"
    "vmla.f32 q11, q7, d6[1] \n"
    "vmla.f32 q12, q6, d7[0] \n"
    "vmla.f32 q13, q7, d7[0] \n"
    "vmla.f32 q14, q6, d7[1] \n"
    "vmla.f32 q15, q7, d7[1] \n"

    "vst1.f32 {d16-d17}, [%[packed_output_data]]! \n"
    "vst1.f32 {d18-d19}, [%[packed_output_data]]! \n"
    "vst1.f32 {d20-d21}, [%[packed_output_data]]! \n"
    "vst1.f32 {d22-d23}, [%[packed_output_data]]! \n"
    "vst1.f32 {d24-d25}, [%[packed_output_data]]! \n"
    "vst1.f32 {d26-d27}, [%[packed_output_data]]! \n"
    "vst1.f32 {d28-d29}, [%[packed_output_data]]! \n"
    "vst1.f32 {d30-d31}, [%[packed_output_data]]! \n"
    :  // outputs
    [lhs_ptr] "+r"(lhs_ptr),
    [rhs_ptr] "+r"(rhs_ptr),
    [packed_output_data] "+r"(packed_output_data),
    [r_depth_block_count] "+r"(r_depth_block_count)
    :  // inputs
    :  // clabbers
    "cc", "memory", "r0",
        "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
  }
#endif
}

void Gemm::PackLhs(const MatrixMap<const float> &lhs,
                   float *packed_lhs) {
#ifdef __aarch64__
  Pack<8, 4>(lhs, ColMajor, packed_lhs);
#else
  Pack<4, 4>(lhs, ColMajor, packed_lhs);
#endif
}

void Gemm::PackRhs(const MatrixMap<const float> &rhs,
                   float *packed_rhs) {
  Pack<8, 4>(rhs, RowMajor, packed_rhs);
}

void Gemm::UnpackOutput(const float *packed_output, MatrixMap<float> *output) {
#ifdef __aarch64__
  Unpack<8, 8>(packed_output, output);
#else
  Unpack<4, 8>(packed_output, output);
#endif
}

template<>
void Gemm::Pack<4, 4>(const MatrixMap<const float> &matrix,
                      MatrixMajor dst_major,
                      float *packed_matrix) {
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
  const float *data = matrix.data();
  float *packed_ptr = packed_matrix;

  const index_t block_size = 4;
  const index_t depth_padded = RoundUp(depth, static_cast<index_t>(4));

  if (depth_padded > depth) {
    memset(packed_ptr + depth * block_size,
           0,
           sizeof(float) * (depth_padded - depth) * block_size);
  }

  if (dst_major == matrix.matrix_major()) {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        memcpy(packed_ptr, data, sizeof(float) * width);
        memset(packed_ptr + width, 0, sizeof(float) * width_remain);
        data += depth_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t d = 0; d < depth; ++d) {
        float32x4_t vi = vld1q_f32(data);
        vst1q_f32(packed_ptr, vi);
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
        memset(packed_ptr + width, 0, sizeof(float) * width_remain);
        packed_ptr += block_size;
      }  // d
    } else {
      const float *data0 = data;
      const float *data1 = data + width_stride;
      const float *data2 = data1 + width_stride;
      const float *data3 = data2 + width_stride;

      const index_t depth_block = depth / 4;
      const index_t depth_remain = depth - depth_block * 4;
      for (index_t depth_block_idx = 0; depth_block_idx < depth_block;
           ++depth_block_idx) {
        float32x4_t v0 = vld1q_f32(data0);
        float32x4_t v1 = vld1q_f32(data1);
        float32x4_t v2 = vld1q_f32(data2);
        float32x4_t v3 = vld1q_f32(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        vst1q_f32(packed_ptr, v0123_intertwined.val[0]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v0123_intertwined.val[1]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v0123n_intertwined.val[0]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v0123n_intertwined.val[1]);
        packed_ptr += 4;

        data0 += 4;
        data1 += 4;
        data2 += 4;
        data3 += 4;
      }
      for (index_t d = 0; d < depth_remain; ++d) {
        float32x4_t vi = {*data0, *data1, *data2, *data3};
        vst1q_f32(packed_ptr, vi);
        packed_ptr += 4;

        ++data0;
        ++data1;
        ++data2;
        ++data3;
      }  // d
    }
  }
}

template<>
void Gemm::Pack<8, 4>(const MatrixMap<const float> &matrix,
                      MatrixMajor dst_major,
                      float *packed_matrix) {
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
  const float *data = matrix.data();
  float *packed_ptr = packed_matrix;

  const index_t block_size = 8;
  const index_t depth_padded = RoundUp(depth, static_cast<index_t>(4));

  if (depth_padded > depth) {
    memset(packed_ptr + depth * block_size,
           0,
           sizeof(float) * (depth_padded - depth) * block_size);
  }

  if (dst_major == matrix.matrix_major()) {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        memcpy(packed_ptr, data, sizeof(float) * width);
        memset(packed_ptr + width, 0, sizeof(float) * width_remain);
        data += depth_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t d = 0; d < depth; ++d) {
        float32x4_t vi = vld1q_f32(data);
        vst1q_f32(packed_ptr, vi);
        float32x4_t vin = vld1q_f32(data + 4);
        vst1q_f32(packed_ptr + 4, vin);
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
        memset(packed_ptr + width, 0, sizeof(float) * width_remain);
        packed_ptr += block_size;
      }  // d
    } else {
      const float *data0 = data;
      const float *data1 = data + width_stride;
      const float *data2 = data1 + width_stride;
      const float *data3 = data2 + width_stride;
      const float *data4 = data3 + width_stride;
      const float *data5 = data4 + width_stride;
      const float *data6 = data5 + width_stride;
      const float *data7 = data6 + width_stride;

      const index_t depth_block = depth / 4;
      const index_t depth_remain = depth - depth_block * 4;
      for (index_t depth_block_idx = 0; depth_block_idx < depth_block;
           ++depth_block_idx) {
        float32x4_t v0 = vld1q_f32(data0);
        float32x4_t v1 = vld1q_f32(data1);
        float32x4_t v2 = vld1q_f32(data2);
        float32x4_t v3 = vld1q_f32(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        float32x4_t v4 = vld1q_f32(data4);
        float32x4_t v5 = vld1q_f32(data5);
        float32x4_t v6 = vld1q_f32(data6);
        float32x4_t v7 = vld1q_f32(data7);
        float32x4x2_t v46_intertwined = vzipq_f32(v4, v6);
        float32x4x2_t v57_intertwined = vzipq_f32(v5, v7);
        float32x4x2_t v4567_intertwined =
            vzipq_f32(v46_intertwined.val[0], v57_intertwined.val[0]);
        float32x4x2_t v4567n_intertwined =
            vzipq_f32(v46_intertwined.val[1], v57_intertwined.val[1]);

        vst1q_f32(packed_ptr, v0123_intertwined.val[0]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v4567_intertwined.val[0]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v0123_intertwined.val[1]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v4567_intertwined.val[1]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v0123n_intertwined.val[0]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v4567n_intertwined.val[0]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v0123n_intertwined.val[1]);
        packed_ptr += 4;

        vst1q_f32(packed_ptr, v4567n_intertwined.val[1]);
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
        vst1q_f32(packed_ptr, vi);
        packed_ptr += 4;

        float32x4_t vin = {*data4, *data5, *data6, *data7};
        vst1q_f32(packed_ptr, vin);
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

template<>
void Gemm::Unpack<4, 8>(const float *packed_output, MatrixMap<float> *output) {
  const index_t rows = output->rows();
  const index_t cols = output->cols();
  index_t row_stride = output->rows_stride();
  index_t col_stride = output->cols_stride();

  float *output_ptr = output->data();
  const float *packed_ptr = packed_output;

  const index_t block_size = 8;

  // packed_output always has row-major
  if (output->matrix_major() == RowMajor) {
    if (cols < block_size) {
      for (index_t r = 0; r < rows; ++r) {
        memcpy(output_ptr, packed_ptr, sizeof(float) * cols);
        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t r = 0; r < rows; ++r) {
        float32x4_t vi = vld1q_f32(packed_ptr);
        vst1q_f32(output_ptr, vi);
        float32x4_t vin = vld1q_f32(packed_ptr + 4);
        vst1q_f32(output_ptr + 4, vin);

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
      const float *data0 = packed_ptr;
      const float *data1 = data0 + block_size;
      const float *data2 = data1 + block_size;
      const float *data3 = data2 + block_size;

      index_t col_block = cols / 4;
      index_t col_remain = cols - col_block * 4;
      for (index_t col_block_idx = 0; col_block_idx < col_block;
           ++col_block_idx) {
        float32x4_t v0 = vld1q_f32(data0);
        float32x4_t v1 = vld1q_f32(data1);
        float32x4_t v2 = vld1q_f32(data2);
        float32x4_t v3 = vld1q_f32(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        vst1q_f32(output_ptr, v0123_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q_f32(output_ptr, v0123_intertwined.val[1]);
        output_ptr += col_stride;

        vst1q_f32(output_ptr, v0123n_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q_f32(output_ptr, v0123n_intertwined.val[1]);
        output_ptr += col_stride;

        data0 += 4;
        data1 += 4;
        data2 += 4;
        data3 += 4;
      }
      for (index_t c = 0; c < col_remain; ++c) {
        float32x4_t vi = {*data0, *data1, *data2, *data3};
        vst1q_f32(output_ptr, vi);
        output_ptr += col_stride;

        ++data0;
        ++data1;
        ++data2;
        ++data3;
      }  // d
    }
  }
}

template<>
void Gemm::Unpack<8, 8>(const float *packed_output, MatrixMap<float> *output) {
  const index_t rows = output->rows();
  const index_t cols = output->cols();
  index_t row_stride = output->rows_stride();
  index_t col_stride = output->cols_stride();

  float *output_ptr = output->data();
  const float *packed_ptr = packed_output;

  const index_t block_size = 8;

  // packed_output always has row-major
  if (output->matrix_major() == RowMajor) {
    if (cols < block_size) {
      for (index_t r = 0; r < rows; ++r) {
        memcpy(output_ptr, packed_ptr, sizeof(float) * cols);
        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t r = 0; r < rows; ++r) {
        float32x4_t vi = vld1q_f32(packed_ptr);
        vst1q_f32(output_ptr, vi);
        float32x4_t vin = vld1q_f32(packed_ptr + 4);
        vst1q_f32(output_ptr + 4, vin);

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
      const float *data0 = packed_ptr;
      const float *data1 = data0 + block_size;
      const float *data2 = data1 + block_size;
      const float *data3 = data2 + block_size;
      const float *data4 = data3 + block_size;
      const float *data5 = data4 + block_size;
      const float *data6 = data5 + block_size;
      const float *data7 = data6 + block_size;

      index_t col_block = cols / 4;
      index_t col_remain = cols - col_block * 4;
      for (index_t col_block_idx = 0; col_block_idx < col_block;
           ++col_block_idx) {
        float32x4_t v0 = vld1q_f32(data0);
        float32x4_t v1 = vld1q_f32(data1);
        float32x4_t v2 = vld1q_f32(data2);
        float32x4_t v3 = vld1q_f32(data3);
        float32x4x2_t v02_intertwined = vzipq_f32(v0, v2);
        float32x4x2_t v13_intertwined = vzipq_f32(v1, v3);
        float32x4x2_t v0123_intertwined =
            vzipq_f32(v02_intertwined.val[0], v13_intertwined.val[0]);
        float32x4x2_t v0123n_intertwined =
            vzipq_f32(v02_intertwined.val[1], v13_intertwined.val[1]);

        float32x4_t v4 = vld1q_f32(data4);
        float32x4_t v5 = vld1q_f32(data5);
        float32x4_t v6 = vld1q_f32(data6);
        float32x4_t v7 = vld1q_f32(data7);
        float32x4x2_t v46_intertwined = vzipq_f32(v4, v6);
        float32x4x2_t v57_intertwined = vzipq_f32(v5, v7);
        float32x4x2_t v4567_intertwined =
            vzipq_f32(v46_intertwined.val[0], v57_intertwined.val[0]);
        float32x4x2_t v4567n_intertwined =
            vzipq_f32(v46_intertwined.val[1], v57_intertwined.val[1]);

        vst1q_f32(output_ptr, v0123_intertwined.val[0]);
        vst1q_f32(output_ptr + 4, v4567_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q_f32(output_ptr, v0123_intertwined.val[1]);
        vst1q_f32(output_ptr + 4, v4567_intertwined.val[1]);
        output_ptr += col_stride;

        vst1q_f32(output_ptr, v0123n_intertwined.val[0]);
        vst1q_f32(output_ptr + 4, v4567n_intertwined.val[0]);
        output_ptr += col_stride;

        vst1q_f32(output_ptr, v0123n_intertwined.val[1]);
        vst1q_f32(output_ptr + 4, v4567n_intertwined.val[1]);
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
        vst1q_f32(output_ptr, vi);
        float32x4_t vin = {*data4, *data5, *data6, *data7};
        vst1q_f32(output_ptr + 4, vin);
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

MaceStatus Gemm::Compute(const OpContext *context,
                         const Tensor *lhs,
                         const Tensor *rhs,
                         const index_t batch,
                         const index_t lhs_rows,
                         const index_t lhs_cols,
                         const index_t rhs_rows,
                         const index_t rhs_cols,
                         const bool transpose_lhs,
                         const bool transpose_rhs,
                         const bool transpose_out,
                         const bool lhs_batched,
                         const bool rhs_batched,
                         Tensor *output) {
  index_t rows = transpose_lhs ? lhs_cols : lhs_rows;
  index_t depth = transpose_lhs ? lhs_rows : lhs_cols;
  index_t cols = transpose_rhs ? rhs_rows : rhs_cols;
  index_t depth2 = transpose_rhs ? rhs_cols : rhs_rows;
  MACE_CHECK(depth == depth2,
             "Matrices that multiply have inconsistent depth dim: ",
             depth,
             " vs. ",
             depth2);

  return Compute(context,
                 lhs,
                 rhs,
                 batch,
                 rows,
                 cols,
                 depth,
                 transpose_lhs ? ColMajor : RowMajor,
                 transpose_rhs ? ColMajor : RowMajor,
                 transpose_out ? ColMajor : RowMajor,
                 lhs_batched,
                 rhs_batched,
                 output);
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
