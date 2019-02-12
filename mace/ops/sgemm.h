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

// This implementation is deprecated. use mace/ops/arm/fp32/gemm.h instead.

#ifndef MACE_OPS_SGEMM_H_
#define MACE_OPS_SGEMM_H_

#include <memory>
#include <utility>

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"
#include "mace/core/allocator.h"
#include "mace/core/tensor.h"

namespace mace {
namespace ops {

enum Major {
  SGemmRowMajor,
  SGemmColMajor
};

template<typename T>
class SGemmMatrixMap {
 public:
  SGemmMatrixMap() {}

  SGemmMatrixMap(const index_t batch,
            const index_t row,
            const index_t col,
            const Major major,
            T *data,
            const bool is_const = false) :
      batch_(batch),
      row_(row),
      col_(col),
      stride_(major == SGemmRowMajor ? col : row),
      major_(major),
      data_(data),
      is_const_(is_const) {}

  SGemmMatrixMap transpose() const {
    Major transpose_major =
        major_ == SGemmRowMajor ? SGemmColMajor : SGemmRowMajor;
    return SGemmMatrixMap(batch_,
                          col_,
                          row_,
                          transpose_major,
                          data_,
                          is_const_);
  }

  index_t batch() const {
    return batch_;
  }

  index_t row() const {
    return row_;
  }

  index_t col() const {
    return col_;
  }

  index_t stride() const {
    return stride_;
  }

  Major map_major() const {
    return major_;
  }

  T *data() const {
    return data_;
  }

  T *batch_data(index_t batch) const {
    return data_ + batch * row_ * col_;
  }

  index_t size() const {
    return batch_ * row_ * col_;
  }

  bool is_const() const {
    return is_const_;
  }

 private:
  index_t batch_;
  index_t row_;
  index_t col_;
  index_t stride_;
  Major major_;
  T *data_;
  bool is_const_;
};

typedef Major PackOrder;
typedef Tensor PackedBlock;

class SGemm {
 public:
  SGemm()
      : packed_lhs_(nullptr),
        packed_rhs_(nullptr),
        packed_(false) {}

  void operator()(const SGemmMatrixMap<const float> &lhs,
                  const SGemmMatrixMap<const float> &rhs,
                  SGemmMatrixMap<float> *result,
                  ScratchBuffer *scratch_buffer = nullptr);

  void Run(const float *A,
           const float *B,
           const index_t batch,
           const index_t height_a,
           const index_t width_a,
           const index_t height_b,
           const index_t width_b,
           const bool transpose_a,
           const bool transpose_b,
           const bool is_a_weight,
           const bool is_b_weight,
           float *C,
           ScratchBuffer *scratch_buffer = nullptr);

  void PackLhs(const SGemmMatrixMap<const float> &lhs,
               PackedBlock *packed_block);

  void PackRhs(const SGemmMatrixMap<const float> &rhs,
               PackedBlock *packed_block);

  void UnPack(const PackedBlock &packed_result,
              SGemmMatrixMap<float> *matrix_map);

 private:
  void Pack(const SGemmMatrixMap<const float> &src,
            const PackOrder order,
            PackedBlock *packed_block);

  void PackPerBatch(const SGemmMatrixMap<const float> &src,
                    const PackOrder order,
                    const index_t batch_index,
                    float *packed_data);

  void UnPackPerBatch(const float *packed_data,
                      const index_t batch_index,
                      SGemmMatrixMap<float> *matrix_map);

  void RunInternal(const PackedBlock &lhs,
                   const PackedBlock &rhs,
                   const index_t batch,
                   const index_t height,
                   const index_t depth,
                   const index_t width,
                   PackedBlock *result);

  void RunPerBatch(const float *lhs,
                   const float *rhs,
                   const index_t height,
                   const index_t depth,
                   const index_t width,
                   float *result);

  std::unique_ptr<Tensor> packed_lhs_;
  std::unique_ptr<Tensor> packed_rhs_;
  std::unique_ptr<Tensor> packed_result_;

  bool packed_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SGEMM_H_
