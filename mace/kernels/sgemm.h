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

#ifndef MACE_KERNELS_SGEMM_H_
#define MACE_KERNELS_SGEMM_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"
#include "mace/core/allocator.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

enum Major {
  RowMajor,
  ColMajor
};

template<typename T>
class MatrixMap {
 public:
  MatrixMap(const index_t row,
            const index_t col,
            const Major major,
            T *data) :
      row_(row),
      col_(col),
      stride_(major == RowMajor ? col : row),
      major_(major),
      data_(data) {}

  MatrixMap<T> transpose(const MatrixMap<T> &matrix_map) {
    Major transpose_major = matrix_map.major_ == RowMajor ? ColMajor : RowMajor;
    return MatrixMap<T>(matrix_map.col_,
                        matrix_map.row_,
                        transpose_major,
                        matrix_map.data_);
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

  Major major() const {
    return major_;
  }

  T *data() const {
    return data_;
  }

  T *data(int row, int col) const {
    return data_ + row * stride_ + col;
  }

 private:
  index_t row_;
  index_t col_;
  index_t stride_;
  Major major_;
  T *data_;
};

typedef Major PackOrder;

template<typename T>
class PackedBlock {
 public:
  PackedBlock() : data_tensor_(GetDeviceAllocator(CPU),
                               DataTypeToEnum<T>::v()) {}

  const T *data() {
    return data_tensor_.data<T>();
  }

  T *mutable_data() {
    return data_tensor_.mutable_data<T>();
  }

  Tensor *tensor() {
    return &data_tensor_;
  }

 private:
  Tensor data_tensor_;
};

class SGemm {
 public:
  void operator()(const MatrixMap<float> &lhs,
                  const MatrixMap<float> &rhs,
                  MatrixMap<float> *result);

  void operator()(const PackedBlock<float> &lhs,
                  const PackedBlock<float> &rhs,
                  const index_t height,
                  const index_t depth,
                  const index_t width,
                  PackedBlock<float> *result);

  void PackLhs(const MatrixMap<float> &lhs, PackedBlock<float> *packed_block);

  void PackRhs(const MatrixMap<float> &rhs, PackedBlock<float> *packed_block);

  void UnPack(const PackedBlock<float> &packed_result,
              MatrixMap<float> *matrix_map);

 private:
  void Pack(const MatrixMap<float> &src,
            const PackOrder order,
            PackedBlock<float> *packed_block);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SGEMM_H_
