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


#ifndef MICRO_OPS_UTILS_MATRIX_H_
#define MICRO_OPS_UTILS_MATRIX_H_

#include "micro/base/logging.h"

namespace micro {
namespace ops {

enum MatrixMajor {
  RowMajor,
  ColMajor
};

inline MatrixMajor TransposeMatrixMajor(const MatrixMajor src_major) {
  return src_major == RowMajor ? ColMajor : RowMajor;
}

template<typename T>
class MatrixMap {
 public:
  MatrixMap()
      : data_(NULL),
        matrix_major_(RowMajor),
        rows_(0),
        cols_(0),
        stride_(0) {}
  MatrixMap(T *data,
            const MatrixMajor matrix_major,
            const int32_t rows,
            const int32_t cols) :
      data_(data),
      matrix_major_(matrix_major),
      rows_(rows),
      cols_(cols),
      stride_(matrix_major == ColMajor ? rows : cols) {}
  MatrixMap(T *data,
            const MatrixMajor matrix_major,
            const int32_t rows,
            const int32_t cols,
            const int32_t stride) :
      data_(data),
      matrix_major_(matrix_major),
      rows_(rows),
      cols_(cols),
      stride_(stride) {}
  MatrixMap(const MatrixMap &other)
      : data_(other.data_),
        matrix_major_(other.matrix_major_),
        rows_(other.rows_),
        cols_(other.cols_),
        stride_(other.stride_) {}

  MatrixMajor matrix_major() const { return matrix_major_; }
  int32_t rows() const { return rows_; }
  int32_t cols() const { return cols_; }
  int32_t stride() const { return stride_; }
  int32_t rows_stride() const {
    return matrix_major_ == ColMajor ? 1 : stride_;
  }
  int32_t cols_stride() const {
    return matrix_major_ == RowMajor ? 1 : stride_;
  }
  int32_t size() const { return rows_ * cols_; }
  T *data() const { return data_; }
  T *data(int32_t rows, int32_t cols) const {
    return data_ + rows * rows_stride() + cols * cols_stride();
  }
  T &operator()(int32_t row, int32_t col) const { return *data(row, col); }
  MatrixMap block(int32_t start_row, int32_t start_col, int32_t block_rows,
                  int32_t block_cols) const {
    MACE_ASSERT(start_row >= 0);
    MACE_ASSERT(start_row + block_rows <= rows_);
    MACE_ASSERT(start_col >= 0);
    MACE_ASSERT(start_col + block_cols <= cols_);

    return MatrixMap(data(start_row, start_col),
                     matrix_major_,
                     block_rows,
                     block_cols,
                     stride_);
  }

 private:
  T *data_;
  MatrixMajor matrix_major_;
  int32_t rows_;
  int32_t cols_;
  int32_t stride_;
};

}  // namespace ops
}  // namespace micro

#endif  // MICRO_OPS_UTILS_MATRIX_H_
