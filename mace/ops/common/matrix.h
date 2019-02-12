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


#ifndef MACE_OPS_COMMON_MATRIX_H_
#define MACE_OPS_COMMON_MATRIX_H_

namespace mace {
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
      : data_(nullptr),
        matrix_major_(RowMajor),
        rows_(0),
        cols_(0),
        stride_(0) {}
  MatrixMap(T *data,
            const MatrixMajor matrix_major,
            const index_t rows,
            const index_t cols) :
      data_(data),
      matrix_major_(matrix_major),
      rows_(rows),
      cols_(cols),
      stride_(matrix_major == ColMajor ? rows : cols) {}
  MatrixMap(T *data,
            const MatrixMajor matrix_major,
            const index_t rows,
            const index_t cols,
            const index_t stride) :
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
  index_t rows() const { return rows_; }
  index_t cols() const { return cols_; }
  index_t stride() const { return stride_; }
  int rows_stride() const {
    return matrix_major_ == MatrixMajor::ColMajor ? 1 : stride_;
  }
  int cols_stride() const {
    return matrix_major_ == MatrixMajor::RowMajor ? 1 : stride_;
  }
  index_t size() const { return rows_ * cols_; }
  T *data() const { return data_; }
  T *data(int rows, int cols) const {
    return data_ + rows * rows_stride() + cols * cols_stride();
  }
  T &operator()(int row, int col) const { return *data(row, col); }
  MatrixMap block(int start_row, int start_col, int block_rows,
                  int block_cols) const {
    MACE_CHECK(start_row >= 0);
    MACE_CHECK(start_row + block_rows <= rows_);
    MACE_CHECK(start_col >= 0);
    MACE_CHECK(start_col + block_cols <= cols_);

    return MatrixMap(data(start_row, start_col),
                     matrix_major_,
                     block_rows,
                     block_cols,
                     stride_);
  }

 private:
  T *data_;
  MatrixMajor matrix_major_;
  index_t rows_;
  index_t cols_;
  index_t stride_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_MATRIX_H_
