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

#include <arm_neon.h>
#include <algorithm>
#include <utility>

#include "mace/ops/arm/base/gemm.h"
#include "mace/port/env.h"

namespace mace {
namespace ops {
namespace arm {

template<>
void Gemm<float16_t>::Pack8x4(const MatrixMap<const float16_t> &matrix,
                              MatrixMajor dst_major,
                              float16_t *packed_matrix) {
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
  const float16_t *data = matrix.data();
  float16_t *packed_ptr = packed_matrix;

  const index_t block_size = 8;
  const index_t depth_padded = RoundUp(depth, static_cast<index_t>(8));

  if (depth_padded > depth) {
    memset(packed_ptr + depth * block_size,
           0,
           sizeof(float16_t) * (depth_padded - depth) * block_size);
  }

  if (dst_major == matrix.matrix_major()) {
    if (width < block_size) {
      const index_t width_remain = block_size - width;
      for (index_t d = 0; d < depth; ++d) {
        memcpy(packed_ptr, data, sizeof(float16_t) * width);
        memset(packed_ptr + width, 0, sizeof(float16_t) * width_remain);
        data += depth_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t d = 0; d < depth; ++d) {
        float16x8_t vi = vld1q_f16(data);
        vst1q_f16(packed_ptr, vi);
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
        memset(packed_ptr + width, 0, sizeof(float16_t) * width_remain);
        packed_ptr += block_size;
      }  // d
    } else {
      const float16_t *data0 = data;
      const float16_t *data1 = data + width_stride;
      const float16_t *data2 = data1 + width_stride;
      const float16_t *data3 = data2 + width_stride;
      const float16_t *data4 = data3 + width_stride;
      const float16_t *data5 = data4 + width_stride;
      const float16_t *data6 = data5 + width_stride;
      const float16_t *data7 = data6 + width_stride;

      const index_t depth_block = depth / 8;
      const index_t depth_remain = depth - depth_block * 8;
      for (index_t depth_block_idx = 0; depth_block_idx < depth_block;
           ++depth_block_idx) {
        float16x8_t v0 = vld1q_f16(data0);
        float16x8_t v1 = vld1q_f16(data1);
        float16x8_t v2 = vld1q_f16(data2);
        float16x8_t v3 = vld1q_f16(data3);
        float16x8_t v4 = vld1q_f16(data4);
        float16x8_t v5 = vld1q_f16(data5);
        float16x8_t v6 = vld1q_f16(data6);
        float16x8_t v7 = vld1q_f16(data7);
        float16x8x2_t v02_intertwined = vzipq_f16(v0, v2);
        float16x8x2_t v13_intertwined = vzipq_f16(v1, v3);
        float16x8x2_t v46_intertwined = vzipq_f16(v4, v6);
        float16x8x2_t v57_intertwined = vzipq_f16(v5, v7);
        float16x8x2_t v0246_intertwined =
            vzipq_f16(v02_intertwined.val[0], v46_intertwined.val[0]);
        float16x8x2_t v0246n_intertwined =
            vzipq_f16(v02_intertwined.val[1], v46_intertwined.val[1]);
        float16x8x2_t v1357_intertwined =
            vzipq_f16(v13_intertwined.val[0], v57_intertwined.val[0]);
        float16x8x2_t v1357n_intertwined =
            vzipq_f16(v13_intertwined.val[1], v57_intertwined.val[1]);

        float16x8x2_t v01234567_intertwined =
            vzipq_f16(v0246_intertwined.val[0], v1357_intertwined.val[0]);
        float16x8x2_t v01234567n1_intertwined =
            vzipq_f16(v0246_intertwined.val[1], v1357_intertwined.val[1]);
        float16x8x2_t v01234567n2_intertwined =
            vzipq_f16(v0246n_intertwined.val[0], v1357n_intertwined.val[0]);
        float16x8x2_t v01234567n3_intertwined =
            vzipq_f16(v0246n_intertwined.val[1], v1357n_intertwined.val[1]);

        vst1q_f16(packed_ptr, v01234567_intertwined.val[0]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567_intertwined.val[1]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567n1_intertwined.val[0]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567n1_intertwined.val[1]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567n2_intertwined.val[0]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567n2_intertwined.val[1]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567n3_intertwined.val[0]);
        packed_ptr += 8;
        vst1q_f16(packed_ptr, v01234567n3_intertwined.val[1]);
        packed_ptr += 8;

        data0 += 8;
        data1 += 8;
        data2 += 8;
        data3 += 8;
        data4 += 8;
        data5 += 8;
        data6 += 8;
        data7 += 8;
      }
      for (index_t d = 0; d < depth_remain; ++d) {
        float16x8_t vi =
            {*data0, *data1, *data2, *data3, *data4, *data5, *data6, *data7};
        vst1q_f16(packed_ptr, vi);
        packed_ptr += 8;


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
void Gemm<float16_t>::Unpack8x8(const float16_t *packed_output,
                                MatrixMap<float16_t> *output) {
  const index_t rows = output->rows();
  const index_t cols = output->cols();
  index_t row_stride = output->rows_stride();
  index_t col_stride = output->cols_stride();

  float16_t *output_ptr = output->data();
  const float16_t *packed_ptr = packed_output;

  const index_t block_size = 8;

  // packed_output always has row-major
  if (output->matrix_major() == RowMajor) {
    if (cols < block_size) {
      for (index_t r = 0; r < rows; ++r) {
        memcpy(output_ptr, packed_ptr, sizeof(float16_t) * cols);
        output_ptr += row_stride;
        packed_ptr += block_size;
      }
    } else {
      for (index_t r = 0; r < rows; ++r) {
        float16x8_t vi = vld1q_f16(packed_ptr);
        vst1q_f16(output_ptr, vi);
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
      const float16_t *data0 = packed_ptr;
      const float16_t *data1 = data0 + block_size;
      const float16_t *data2 = data1 + block_size;
      const float16_t *data3 = data2 + block_size;
      const float16_t *data4 = data3 + block_size;
      const float16_t *data5 = data4 + block_size;
      const float16_t *data6 = data5 + block_size;
      const float16_t *data7 = data6 + block_size;

      index_t col_block = cols / 8;
      index_t col_remain = cols - col_block * 8;
      for (index_t col_block_idx = 0; col_block_idx < col_block;
           ++col_block_idx) {
        float16x8_t v0 = vld1q_f16(data0);
        float16x8_t v1 = vld1q_f16(data1);
        float16x8_t v2 = vld1q_f16(data2);
        float16x8_t v3 = vld1q_f16(data3);
        float16x8_t v4 = vld1q_f16(data4);
        float16x8_t v5 = vld1q_f16(data5);
        float16x8_t v6 = vld1q_f16(data6);
        float16x8_t v7 = vld1q_f16(data7);
        float16x8x2_t v02_intertwined = vzipq_f16(v0, v2);
        float16x8x2_t v13_intertwined = vzipq_f16(v1, v3);
        float16x8x2_t v46_intertwined = vzipq_f16(v4, v6);
        float16x8x2_t v57_intertwined = vzipq_f16(v5, v7);
        float16x8x2_t v0246_intertwined =
            vzipq_f16(v02_intertwined.val[0], v46_intertwined.val[0]);
        float16x8x2_t v0246n_intertwined =
            vzipq_f16(v02_intertwined.val[1], v46_intertwined.val[1]);
        float16x8x2_t v1357_intertwined =
            vzipq_f16(v13_intertwined.val[0], v57_intertwined.val[0]);
        float16x8x2_t v1357n_intertwined =
            vzipq_f16(v13_intertwined.val[1], v57_intertwined.val[1]);

        float16x8x2_t v01234567_intertwined =
            vzipq_f16(v0246_intertwined.val[0], v1357_intertwined.val[0]);
        float16x8x2_t v01234567n1_intertwined =
            vzipq_f16(v0246_intertwined.val[1], v1357_intertwined.val[1]);
        float16x8x2_t v01234567n2_intertwined =
            vzipq_f16(v0246n_intertwined.val[0], v1357n_intertwined.val[0]);
        float16x8x2_t v01234567n3_intertwined =
            vzipq_f16(v0246n_intertwined.val[1], v1357n_intertwined.val[1]);

        vst1q_f16(output_ptr, v01234567_intertwined.val[0]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567_intertwined.val[1]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567n1_intertwined.val[0]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567n1_intertwined.val[1]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567n2_intertwined.val[0]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567n2_intertwined.val[1]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567n3_intertwined.val[0]);
        output_ptr += col_stride;
        vst1q_f16(output_ptr, v01234567n3_intertwined.val[1]);
        output_ptr += col_stride;

        data0 += 8;
        data1 += 8;
        data2 += 8;
        data3 += 8;
        data4 += 8;
        data5 += 8;
        data6 += 8;
        data7 += 8;
      }
      for (index_t c = 0; c < col_remain; ++c) {
        float16x8_t vi =
            {*data0, *data1, *data2, *data3, *data4, *data5, *data6, *data7};
        vst1q_f16(output_ptr, vi);
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

template<>
void Gemm<float16_t>::PackLhs(const MatrixMap<const float16_t> &lhs,
                              float16_t *packed_lhs) {
  Pack8x4(lhs, ColMajor, packed_lhs);
}

template<>
void Gemm<float16_t>::PackRhs(const MatrixMap<const float16_t> &rhs,
                              float16_t *packed_rhs) {
  Pack8x4(rhs, RowMajor, packed_rhs);
}

template<>
void Gemm<float16_t>::UnpackOutput(const float16_t *packed_output,
                                   MatrixMap<float16_t> *output) {
  Unpack8x8(packed_output, output);
}

template<>
void Gemm<float16_t>::ComputeBlock(const float16_t *packed_lhs_data,
                                   const float16_t *packed_rhs_data,
                                   const index_t depth_padded,
                                   float16_t *packed_output_data) {
  /* Ref:
  for (index_t r = 0; r < block_size; ++r) {
    for (index_t c = 0; c < block_size; ++c) {
      float16_t sum = 0;
      for (index_t d = 0; d < depth; ++d) {
        // (r, d) * (d, c)
        sum += packed_lhs_data[d * r_block_size + r]
            * packed_rhs_data[d * c_block_size + c];
      }
      packed_output_data[r * c_block_size + c] = sum;
    }
  }
  */
  const float16_t *lhs_ptr = packed_lhs_data;
  const float16_t *rhs_ptr = packed_rhs_data;

  const index_t depth_block_count = depth_padded / 8;

  if (depth_block_count > 0) {
    index_t r_depth_block_count = depth_block_count;
    // just make compiler happy
    MACE_UNUSED(r_depth_block_count);

    asm volatile(
        "dup v16.8h, wzr \n"
        "dup v17.8h, wzr \n"
        "dup v18.8h, wzr \n"
        "dup v19.8h, wzr \n"
        "dup v20.8h, wzr \n"
        "dup v21.8h, wzr \n"
        "dup v22.8h, wzr \n"
        "dup v23.8h, wzr \n"
        "dup v24.8h, wzr \n"
        "dup v25.8h, wzr \n"
        "dup v26.8h, wzr \n"
        "dup v27.8h, wzr \n"
        "dup v28.8h, wzr \n"
        "dup v29.8h, wzr \n"
        "dup v30.8h, wzr \n"
        "dup v31.8h, wzr \n"

        // prelogue
        "ld1 {v0.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v1.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v2.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v3.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v4.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v5.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v6.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v7.8h}, [%[lhs_ptr]], #16 \n"

        "ld1 {v8.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v9.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v10.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v11.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v12.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v13.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v14.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v15.8h}, [%[rhs_ptr]], #16 \n"

        "subs %[r_depth_block_count], %[r_depth_block_count], #1 \n"
        "beq 1f\n"

        "0: \n"
        "fmla v16.8h, v8.8h, v0.h[0] \n"
        "fmla v17.8h, v8.8h, v0.h[1] \n"
        "fmla v18.8h, v8.8h, v0.h[2] \n"
        "fmla v19.8h, v8.8h, v0.h[3] \n"
        "fmla v20.8h, v8.8h, v0.h[4] \n"
        "fmla v21.8h, v8.8h, v0.h[5] \n"
        "fmla v22.8h, v8.8h, v0.h[6] \n"
        "fmla v23.8h, v8.8h, v0.h[7] \n"

        "ld1 {v0.8h}, [%[lhs_ptr]], #16 \n"

        "fmla v24.8h, v9.8h, v1.h[0] \n"
        "fmla v25.8h, v9.8h, v1.h[1] \n"
        "fmla v26.8h, v9.8h, v1.h[2] \n"
        "fmla v27.8h, v9.8h, v1.h[3] \n"
        "fmla v28.8h, v9.8h, v1.h[4] \n"
        "fmla v29.8h, v9.8h, v1.h[5] \n"
        "fmla v30.8h, v9.8h, v1.h[6] \n"
        "fmla v31.8h, v9.8h, v1.h[7] \n"

        "ld1 {v1.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v8.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v9.8h}, [%[rhs_ptr]], #16 \n"

        "fmla v16.8h, v10.8h, v2.h[0] \n"
        "fmla v17.8h, v10.8h, v2.h[1] \n"
        "fmla v18.8h, v10.8h, v2.h[2] \n"
        "fmla v19.8h, v10.8h, v2.h[3] \n"
        "fmla v20.8h, v10.8h, v2.h[4] \n"
        "fmla v21.8h, v10.8h, v2.h[5] \n"
        "fmla v22.8h, v10.8h, v2.h[6] \n"
        "fmla v23.8h, v10.8h, v2.h[7] \n"

        "ld1 {v2.8h}, [%[lhs_ptr]], #16 \n"

        "fmla v24.8h, v11.8h, v3.h[0] \n"
        "fmla v25.8h, v11.8h, v3.h[1] \n"
        "fmla v26.8h, v11.8h, v3.h[2] \n"
        "fmla v27.8h, v11.8h, v3.h[3] \n"
        "fmla v28.8h, v11.8h, v3.h[4] \n"
        "fmla v29.8h, v11.8h, v3.h[5] \n"
        "fmla v30.8h, v11.8h, v3.h[6] \n"
        "fmla v31.8h, v11.8h, v3.h[7] \n"

        "ld1 {v3.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v10.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v11.8h}, [%[rhs_ptr]], #16 \n"

        "fmla v16.8h, v12.8h, v4.h[0] \n"
        "fmla v17.8h, v12.8h, v4.h[1] \n"
        "fmla v18.8h, v12.8h, v4.h[2] \n"
        "fmla v19.8h, v12.8h, v4.h[3] \n"
        "fmla v20.8h, v12.8h, v4.h[4] \n"
        "fmla v21.8h, v12.8h, v4.h[5] \n"
        "fmla v22.8h, v12.8h, v4.h[6] \n"
        "fmla v23.8h, v12.8h, v4.h[7] \n"
        "ld1 {v4.8h}, [%[lhs_ptr]], #16 \n"

        "fmla v24.8h, v13.8h, v5.h[0] \n"
        "fmla v25.8h, v13.8h, v5.h[1] \n"
        "fmla v26.8h, v13.8h, v5.h[2] \n"
        "fmla v27.8h, v13.8h, v5.h[3] \n"
        "fmla v28.8h, v13.8h, v5.h[4] \n"
        "fmla v29.8h, v13.8h, v5.h[5] \n"
        "fmla v30.8h, v13.8h, v5.h[6] \n"
        "fmla v31.8h, v13.8h, v5.h[7] \n"

        "ld1 {v5.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v12.8h}, [%[rhs_ptr]], #16 \n"
        "ld1 {v13.8h}, [%[rhs_ptr]], #16 \n"

        "fmla v16.8h, v14.8h, v6.h[0] \n"
        "fmla v17.8h, v14.8h, v6.h[1] \n"
        "fmla v18.8h, v14.8h, v6.h[2] \n"
        "fmla v19.8h, v14.8h, v6.h[3] \n"
        "fmla v20.8h, v14.8h, v6.h[4] \n"
        "fmla v21.8h, v14.8h, v6.h[5] \n"
        "fmla v22.8h, v14.8h, v6.h[6] \n"
        "fmla v23.8h, v14.8h, v6.h[7] \n"

        "ld1 {v6.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v14.8h}, [%[rhs_ptr]], #16 \n"

        "subs %[r_depth_block_count], %[r_depth_block_count], #1 \n"

        "fmla v24.8h, v15.8h, v7.h[0] \n"
        "fmla v25.8h, v15.8h, v7.h[1] \n"
        "fmla v26.8h, v15.8h, v7.h[2] \n"
        "fmla v27.8h, v15.8h, v7.h[3] \n"
        "fmla v28.8h, v15.8h, v7.h[4] \n"
        "fmla v29.8h, v15.8h, v7.h[5] \n"
        "fmla v30.8h, v15.8h, v7.h[6] \n"
        "fmla v31.8h, v15.8h, v7.h[7] \n"

        "ld1 {v7.8h}, [%[lhs_ptr]], #16 \n"
        "ld1 {v15.8h}, [%[rhs_ptr]], #16 \n"

        "bne 0b \n"

        // prologue
        "1:\n"
        "fmla v16.8h, v8.8h, v0.h[0] \n"
        "fmla v17.8h, v8.8h, v0.h[1] \n"
        "fmla v18.8h, v8.8h, v0.h[2] \n"
        "fmla v19.8h, v8.8h, v0.h[3] \n"
        "fmla v20.8h, v8.8h, v0.h[4] \n"
        "fmla v21.8h, v8.8h, v0.h[5] \n"
        "fmla v22.8h, v8.8h, v0.h[6] \n"
        "fmla v23.8h, v8.8h, v0.h[7] \n"

        "fmla v24.8h, v9.8h, v1.h[0] \n"
        "fmla v25.8h, v9.8h, v1.h[1] \n"
        "fmla v26.8h, v9.8h, v1.h[2] \n"
        "fmla v27.8h, v9.8h, v1.h[3] \n"
        "fmla v28.8h, v9.8h, v1.h[4] \n"
        "fmla v29.8h, v9.8h, v1.h[5] \n"
        "fmla v30.8h, v9.8h, v1.h[6] \n"
        "fmla v31.8h, v9.8h, v1.h[7] \n"

        "fmla v16.8h, v10.8h, v2.h[0] \n"
        "fmla v17.8h, v10.8h, v2.h[1] \n"
        "fmla v18.8h, v10.8h, v2.h[2] \n"
        "fmla v19.8h, v10.8h, v2.h[3] \n"
        "fmla v20.8h, v10.8h, v2.h[4] \n"
        "fmla v21.8h, v10.8h, v2.h[5] \n"
        "fmla v22.8h, v10.8h, v2.h[6] \n"
        "fmla v23.8h, v10.8h, v2.h[7] \n"

        "fmla v24.8h, v11.8h, v3.h[0] \n"
        "fmla v25.8h, v11.8h, v3.h[1] \n"
        "fmla v26.8h, v11.8h, v3.h[2] \n"
        "fmla v27.8h, v11.8h, v3.h[3] \n"
        "fmla v28.8h, v11.8h, v3.h[4] \n"
        "fmla v29.8h, v11.8h, v3.h[5] \n"
        "fmla v30.8h, v11.8h, v3.h[6] \n"
        "fmla v31.8h, v11.8h, v3.h[7] \n"

        "fmla v16.8h, v12.8h, v4.h[0] \n"
        "fmla v17.8h, v12.8h, v4.h[1] \n"
        "fmla v18.8h, v12.8h, v4.h[2] \n"
        "fmla v19.8h, v12.8h, v4.h[3] \n"
        "fmla v20.8h, v12.8h, v4.h[4] \n"
        "fmla v21.8h, v12.8h, v4.h[5] \n"
        "fmla v22.8h, v12.8h, v4.h[6] \n"
        "fmla v23.8h, v12.8h, v4.h[7] \n"

        "fmla v24.8h, v13.8h, v5.h[0] \n"
        "fmla v25.8h, v13.8h, v5.h[1] \n"
        "fmla v26.8h, v13.8h, v5.h[2] \n"
        "fmla v27.8h, v13.8h, v5.h[3] \n"
        "fmla v28.8h, v13.8h, v5.h[4] \n"
        "fmla v29.8h, v13.8h, v5.h[5] \n"
        "fmla v30.8h, v13.8h, v5.h[6] \n"
        "fmla v31.8h, v13.8h, v5.h[7] \n"

        "fmla v16.8h, v14.8h, v6.h[0] \n"
        "fmla v17.8h, v14.8h, v6.h[1] \n"
        "fmla v18.8h, v14.8h, v6.h[2] \n"
        "fmla v19.8h, v14.8h, v6.h[3] \n"
        "fmla v20.8h, v14.8h, v6.h[4] \n"
        "fmla v21.8h, v14.8h, v6.h[5] \n"
        "fmla v22.8h, v14.8h, v6.h[6] \n"
        "fmla v23.8h, v14.8h, v6.h[7] \n"

        "fmla v24.8h, v15.8h, v7.h[0] \n"
        "fmla v25.8h, v15.8h, v7.h[1] \n"
        "fmla v26.8h, v15.8h, v7.h[2] \n"
        "fmla v27.8h, v15.8h, v7.h[3] \n"
        "fmla v28.8h, v15.8h, v7.h[4] \n"
        "fmla v29.8h, v15.8h, v7.h[5] \n"
        "fmla v30.8h, v15.8h, v7.h[6] \n"
        "fmla v31.8h, v15.8h, v7.h[7] \n"

        "st1 {v16.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v17.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v18.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v19.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v20.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v21.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v22.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v23.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v24.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v25.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v26.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v27.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v28.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v29.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v30.8h}, [%[packed_output_data]], #16 \n"
        "st1 {v31.8h}, [%[packed_output_data]], #16 \n"
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
}

template<>
MaceStatus Gemm<float16_t>::Compute(const OpContext *context,
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
  const float16_t *lhs_data = lhs->data<float16_t>();
  const float16_t *rhs_data = rhs->data<float16_t>();
  float16_t *output_data = output->mutable_data<float16_t>();

  const index_t row_block_size = 8;
  const index_t col_block_size = 8;
  const index_t depth_block_size = 8;
  const index_t row_block_count = RoundUpDiv(rows, row_block_size);
  const index_t col_block_count = RoundUpDiv(cols, col_block_size);
  const index_t rows_padded = RoundUp(rows, row_block_size);
  const index_t cols_padded = RoundUp(cols, col_block_size);
  const index_t depth_padded = RoundUp(depth, depth_block_size);

  ScratchBuffer *scratch = context->device()->scratch_buffer();

  index_t packed_lhs_size =
      PadAlignSize(sizeof(float16_t) * rows_padded * depth_padded);
  index_t packed_rhs_size =
      PadAlignSize(sizeof(float16_t) * depth_padded * cols_padded);
  index_t packed_output_size =
      PadAlignSize(sizeof(float16_t) * rows_padded * cols_padded);
  // resize to the total size of lhs & rhs & output anyway,
  // in case we do not cache const tensor for saving memory
  scratch->Rewind();
  MACE_RETURN_IF_ERROR(scratch->GrowSize(
      packed_lhs_size + packed_rhs_size + packed_output_size));
  float16_t *packed_lhs_data =
      scratch->Scratch(packed_lhs_size).mutable_data<float16_t>();
  float16_t *packed_rhs_data =
      scratch->Scratch(packed_rhs_size).mutable_data<float16_t>();
  float16_t *packed_output_data =
      scratch->Scratch(packed_output_size).mutable_data<float16_t>();

  int cache_side = kNoCache;
  if (cached_ == kCacheLhs) {
    packed_lhs_data = pack_cache_.mutable_data<float16_t>();
  } else if (cached_ == kCacheRhs) {
    packed_rhs_data = pack_cache_.mutable_data<float16_t>();
  } else if (should_cache_pack_) {
    if (lhs->is_weight() && (!lhs_batched || batch == 1)) {
      cache_side = kCacheLhs;
      pack_cache_.Resize(packed_lhs_size);
      packed_lhs_data = pack_cache_.mutable_data<float16_t>();
    } else if (rhs->is_weight() && (!rhs_batched || batch == 1)) {
      cache_side = kCacheRhs;
      pack_cache_.Resize(packed_rhs_size);
      packed_rhs_data = pack_cache_.mutable_data<float16_t>();
    }
  }

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  for (index_t b = 0; b < batch; ++b) {
    MatrixMap<const float16_t>
        lhs_matrix
        (lhs_data + static_cast<index_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const float16_t>
        rhs_matrix
        (rhs_data + static_cast<index_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<float16_t> output_matrix
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
          float16_t *packed_lhs_data_block =
              packed_lhs_data + row_block_idx * row_block_size * depth_padded;
          PackLhs(lhs_matrix.block(start_row, 0, row_block_len, depth),
                  packed_lhs_data_block);
        }
      }, 0, row_block_count, 1);

      if (cache_side == kCacheLhs) {
        cached_ = kCacheLhs;
        if (lhs->UnderlyingBuffer()->OnHost()) {
          AdviseFree(reinterpret_cast<void *>(const_cast<float16_t *>(lhs->data<
                     float16_t>())),
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
          float16_t *packed_rhs_data_block =
              packed_rhs_data + col_block_idx * col_block_size * depth_padded;
          PackRhs(rhs_matrix.block(0, start_col, depth, col_block_len),
                  packed_rhs_data_block);
        }
      }, 0, col_block_count, 1);

      if (cache_side == kCacheRhs) {
        cached_ = kCacheRhs;
        if (rhs->UnderlyingBuffer()->OnHost()) {
          AdviseFree(reinterpret_cast<void *>(const_cast<float16_t *>(rhs->data<
                     float16_t>())),
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
        const float16_t *packed_lhs_data_block =
            packed_lhs_data + row_block_idx * row_block_size * depth_padded;

        for (index_t col_block_idx = 0; col_block_idx < col_block_count;
             ++col_block_idx) {
          const index_t start_col = col_block_idx * col_block_size;
          const index_t
              col_block_len = std::min(col_block_size, cols - start_col);
          const float16_t *packed_rhs_data_block =
              packed_rhs_data + col_block_idx * col_block_size * depth_padded;
          float16_t *packed_output_data_block =
              packed_output_data + row_block_idx * row_block_size * cols_padded
                  + col_block_idx * col_block_size;
          ComputeBlock(packed_lhs_data_block,
                       packed_rhs_data_block,
                       depth_padded,
                       packed_output_data_block);
          MatrixMap<float16_t> output_block =
              output_matrix.block(start_row,
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

void RegisterFP16GemmDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_FP16_DELEGATOR(
      registry, Gemm<float16_t>, delegator::GemmParam,
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, float16_t, ImplType::NEON));
}
}  // namespace arm
}  // namespace ops
}  // namespace mace
