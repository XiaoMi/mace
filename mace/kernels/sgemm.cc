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

#include <algorithm>
#include <cstring>
#include <vector>

#include "mace/kernels/sgemm.h"

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

namespace mace {
namespace kernels {

void SGemm::operator()(const MatrixMap<const float> &lhs,
                       const MatrixMap<const float> &rhs,
                       MatrixMap<float> *result) {
  PackedBlock<float> packed_lhs;
  PackLhs(lhs, &packed_lhs);

  PackedBlock<float> packed_rhs;
  PackRhs(rhs, &packed_rhs);

  PackedBlock<float> packed_result;
  operator()(packed_lhs,
             packed_rhs,
             lhs.row(),
             lhs.col(),
             rhs.col(),
             &packed_result);
  UnPack(packed_result, result);
}

void SGemm::operator()(const PackedBlock<float> &lhs,
                       const PackedBlock<float> &rhs,
                       const index_t height,
                       const index_t depth,
                       const index_t width,
                       PackedBlock<float> *result) {
  (void) lhs;
  (void) rhs;
  (void) result;
  (void) height;
  (void) depth;
  (void) width;

  // (8, 8) * (8, 4)

  // (4, 4) * (4, 4)

  // remain
}

void SGemm::PackLhs(const MatrixMap<const float> &lhs,
                    PackedBlock<float> *packed_block) {
  Pack(lhs, PackOrder::ColMajor, packed_block);
}

void SGemm::PackRhs(const MatrixMap<const float> &rhs,
                    PackedBlock<float> *packed_block) {
  Pack(rhs, PackOrder::RowMajor, packed_block);
}

void SGemm::UnPack(const PackedBlock<float> &packed_result,
                   MatrixMap<float> *matrix_map) {
  MACE_CHECK_NOTNULL(matrix_map);

  const index_t height = matrix_map->row();
  const index_t width = matrix_map->col();
  auto packed_data = packed_result.data();
  auto unpacked_data = matrix_map->data();

  if (matrix_map->major() == Major::RowMajor) {
    // This is for non-transposed result
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
    #pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *packed_data_ptr = packed_data + iw * height;
      float *unpacked_data_ptr = unpacked_data + iw;
      for (index_t h = 0; h < height; ++h) {
        const index_t packed_offset = h * 4;
        const index_t unpacked_offset = h * width;
        float32x4_t vs = vld1q_f32(packed_data_ptr + packed_offset);
        vst1q_f32(unpacked_data_ptr + unpacked_offset, vs);
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      const float *packed_data_ptr = packed_data + iw * height;
      float *unpacked_data_ptr = unpacked_data + iw;
      for (index_t h = 0; h < height; ++h) {
        unpacked_data_ptr[h * width] = packed_data_ptr[h];
      }
    }
  } else {
    // This is for transposed result
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *packed_data_ptr = packed_data + iw * height;
      float *unpacked_data_ptr = unpacked_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        const index_t packed_offset = h * 4;
        const index_t unpacked_offset = h;
        float32x4_t vs = vld1q_f32(packed_data_ptr + packed_offset);
        unpacked_data_ptr[unpacked_offset] = vs[0];
        unpacked_data_ptr[unpacked_offset + height] = vs[1];
        unpacked_data_ptr[unpacked_offset + 2 * height] = vs[2];
        unpacked_data_ptr[unpacked_offset + 3 * height] = vs[3];
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      std::copy_n(
          packed_data + iw * height, height, unpacked_data + iw * height);
    }
  }
}

void SGemm::Pack(const MatrixMap<const float> &src,
                 const PackOrder order,
                 PackedBlock<float> *packed_block) {
  MACE_CHECK_NOTNULL(packed_block);

  const index_t height = src.row();
  const index_t width = src.col();
  auto src_data = src.data();
  auto packed_data = packed_block->mutable_data();

  if (src.major() == Major::RowMajor && order == PackOrder::ColMajor) {
    // This is for packing no-transpose lhs.
    index_t h = 0;
#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 8; ih += 8) {
      const float *src_data_ptr = src_data + ih * width;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w;
        const index_t packed_offset = w * 8;
        float32x4_t vs0 = {src_data_ptr[src_offset],
                           src_data_ptr[src_offset + width],
                           src_data_ptr[src_offset + 2 * width],
                           src_data_ptr[src_offset + 3 * width]};
        float32x4_t vs1 = {src_data_ptr[src_offset + 4 * width],
                           src_data_ptr[src_offset + 5 * width],
                           src_data_ptr[src_offset + 6 * width],
                           src_data_ptr[src_offset + 7 * width]};
        vst1q_f32(packed_data_ptr + packed_offset, vs0);
        vst1q_f32(packed_data_ptr + packed_offset + 4, vs1);
      }
    }
    h += (height - h) / 8 * 8;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 4; ih += 4) {
      const float *src_data_ptr = src_data + ih * width;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w;
        const index_t packed_offset = w * 4;
        float32x4_t vs = {src_data_ptr[src_offset],
                          src_data_ptr[src_offset + width],
                          src_data_ptr[src_offset + 2 * width],
                          src_data_ptr[src_offset + 3 * width]};
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    h += (height - h) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih < height; ++ih) {
      std::copy_n(src_data + ih * width, width, packed_data + ih * width);
    }
  } else if (src.major() == Major::ColMajor && order == PackOrder::ColMajor) {
    // This is for packing transpose-needed lhs.
    index_t h = 0;
#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 8; ih += 8) {
      const float *src_data_ptr = src_data + ih;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w * height;
        const index_t packed_offset = w * 8;
        float32x4_t vs0 = vld1q_f32(src_data_ptr + src_offset);
        float32x4_t vs1 = vld1q_f32(src_data_ptr + src_offset + 4);
        vst1q_f32(packed_data_ptr + packed_offset, vs0);
        vst1q_f32(packed_data_ptr + packed_offset + 4, vs1);
      }
    }
    h += (height - h) / 8 * 8;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 4; ih += 4) {
      const float *src_data_ptr = src_data + ih;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w * height;
        const index_t packed_offset = w * 4;
        float32x4_t vs = vld1q_f32(src_data_ptr + src_offset);
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    h += (height - h) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih < height; ++ih) {
      const float *src_data_ptr = src_data + ih;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        packed_data_ptr[w] = src_data_ptr[w * height];
      }
    }
  } else if (src.major() == Major::RowMajor && order == PackOrder::RowMajor) {
    // This is for packing no-transpose rhs.
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *src_data_ptr = src_data + iw;
      float *packed_data_ptr = packed_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        const index_t src_offset = h * width;
        const index_t packed_offset = h * 4;
        float32x4_t vs = vld1q_f32(src_data_ptr + src_offset);
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      const float *src_data_ptr = src_data + iw;
      float *packed_data_ptr = packed_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        packed_data_ptr[h] = src_data_ptr[h * width];
      }
    }
  } else if (src.major() == Major::ColMajor && order == PackOrder::RowMajor) {
    // This is for packing transpose-needed rhs.
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *src_data_ptr = src_data + iw * height;
      float *packed_data_ptr = packed_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        const index_t src_offset = h;
        const index_t packed_offset = h * 4;
        float32x4_t vs = {src_data_ptr[src_offset],
                          src_data_ptr[src_offset + height],
                          src_data_ptr[src_offset + 2 * height],
                          src_data_ptr[src_offset + 3 * height]};
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      std::copy_n(src_data + iw * height, height, packed_data + iw * height);
    }
  }
}

}  // namespace kernels
}  // namespace mace
