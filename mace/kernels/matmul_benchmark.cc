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

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "public/gemmlowp.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/kernels/gemm.h"
#include "mace/kernels/gemmlowp_util.h"

namespace gemmlowp {

template<typename tScalar, MapOrder tOrder>
class Matrix : public MatrixMap<tScalar, tOrder> {
 public:
  typedef MatrixMap<tScalar, tOrder> Map;
  typedef MatrixMap<const tScalar, tOrder> ConstMap;
  typedef typename Map::Scalar Scalar;
  static const MapOrder Order = tOrder;
  using Map::cols_;
  using Map::data_;
  using Map::kOrder;
  using Map::rows_;
  using Map::stride_;

 public:
  Matrix() : Map(nullptr, 0, 0, 0) {}

  Matrix(int rows, int cols) : Map(nullptr, 0, 0, 0) { Resize(rows, cols); }

  Matrix(const Matrix &other) : Map(nullptr, 0, 0, 0) { *this = other; }

  Matrix &operator=(const Matrix &other) {
    Resize(other.rows_, other.cols_);
    std::memcpy(data_, other.data_, size() * sizeof(Scalar));
    return *this;
  }

  friend bool operator==(const Matrix &a, const Matrix &b) {
    return a.rows_ == b.rows_ && a.cols_ == b.cols_ &&
        !std::memcmp(a.data_, b.data_, a.size());
  }

  void Resize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    stride_ = kOrder == gemmlowp::MapOrder::ColMajor ? rows : cols;
    storage.resize(size());
    data_ = storage.data();
  }

  int size() const { return rows_ * cols_; }

  Map &map() { return *static_cast<Map *>(this); }

  ConstMap const_map() const { return ConstMap(data_, rows_, cols_, stride_); }

 protected:
  std::vector<Scalar> storage;
};

template<typename MatrixType>
void MakeZero(MatrixType *m) {
  for (int c = 0; c < m->cols(); c++) {
    for (int r = 0; r < m->rows(); r++) {
      (*m)(r, c) = 128;
    }
  }
}

}  // namespace gemmlowp

namespace mace {
namespace kernels {
namespace test {

// Test the speed of different access order of a NHWC buffer

namespace {

// Matmul with (m, k) x (k, n)
void MatmulBenchmark_Mace(int iters, int m, int k, int n) {
  mace::testing::StopTiming();
  std::vector<float> lhs(m * k);
  std::vector<float> rhs(k * n);
  std::vector<float> result(m * n);
  // warm up
  Gemm(lhs.data(), rhs.data(), 1, m, k, n, result.data());
  mace::testing::StartTiming();
  while (iters--) {
    Gemm(lhs.data(), rhs.data(), 1, m, k, n, result.data());
  }
}

void MatmulBenchmark_Eigen(int iters, int m, int k, int n) {
  mace::testing::StopTiming();
  Eigen::MatrixXf lhs = Eigen::MatrixXf::Random(m, k);
  Eigen::MatrixXf rhs = Eigen::MatrixXf::Random(k, n);
  Eigen::MatrixXf result = Eigen::MatrixXf::Zero(m, n);
  // warm up
  result = lhs * rhs;
  mace::testing::StartTiming();
  while (iters--) {
    result = lhs * rhs;
  }
}

void MatmulBenchmark_gemmlowp_uint8(int iters, int rows, int depth, int cols) {
  mace::testing::StopTiming();

  gemmlowp::Matrix<std::uint8_t, gemmlowp::MapOrder::RowMajor> lhs;
  gemmlowp::Matrix<std::uint8_t, gemmlowp::MapOrder::ColMajor> rhs;
  gemmlowp::Matrix<std::uint8_t, gemmlowp::MapOrder::ColMajor> result;
  lhs.Resize(rows, depth);
  rhs.Resize(depth, cols);
  result.Resize(rows, cols);
  gemmlowp::MakeZero(&lhs);
  gemmlowp::MakeZero(&rhs);
  gemmlowp::MakeZero(&result);

  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = 128;
  quantize_down_stage.result_fixedpoint_multiplier = 1234567890;
  quantize_down_stage.result_shift = 16;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  const auto output_pipeline =
      std::make_tuple(quantize_down_stage, saturating_cast_stage);

  gemmlowp::GemmContext& gemm_context = GetGemmlowpContext();
  using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;

  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t, BitDepthParams>(
      &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
      -128, output_pipeline);

  mace::testing::StartTiming();
  while (iters--) {
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                     BitDepthParams>(
        &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
        -128, output_pipeline);
  }
}

void MatmulBenchmark_gemmlowp_int32(int iters, int rows, int depth, int cols) {
  mace::testing::StopTiming();

  gemmlowp::Matrix<std::uint8_t, gemmlowp::MapOrder::RowMajor> lhs;
  gemmlowp::Matrix<std::uint8_t, gemmlowp::MapOrder::ColMajor> rhs;
  gemmlowp::Matrix<std::int32_t, gemmlowp::MapOrder::ColMajor> result;
  lhs.Resize(rows, depth);
  rhs.Resize(depth, cols);
  result.Resize(rows, cols);
  gemmlowp::MakeZero(&lhs);
  gemmlowp::MakeZero(&rhs);
  gemmlowp::MakeZero(&result);

  const auto output_pipeline = std::make_tuple();

  gemmlowp::GemmContext& gemm_context = GetGemmlowpContext();
  using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;

  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t, BitDepthParams>(
      &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
      -128, output_pipeline);

  mace::testing::StartTiming();
  while (iters--) {
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                     BitDepthParams>(
        &gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
        -128, output_pipeline);
  }
}

}  // namespace

#define MACE_BM_MATMUL_FUNC(M, K, N, FUNC, TYPE)                   \
  static void MACE_BM_MATMUL_##M##_##K##_##N##_##FUNC(int iters) { \
    const int64_t macc = static_cast<int64_t>(iters) * M * K * N;  \
    const int64_t tot = static_cast<int64_t>(iters) * (M + N) * K; \
    mace::testing::MaccProcessed(macc);                            \
    mace::testing::BytesProcessed(tot * sizeof(TYPE));             \
    MatmulBenchmark_##FUNC(iters, M, K, N);                        \
  }                                                                \
  MACE_BENCHMARK(MACE_BM_MATMUL_##M##_##K##_##N##_##FUNC)

#define MACE_BM_MATMUL(M, K, N)                          \
  MACE_BM_MATMUL_FUNC(M, K, N, Mace, float);             \
  MACE_BM_MATMUL_FUNC(M, K, N, Eigen, float);            \
  MACE_BM_MATMUL_FUNC(M, K, N, gemmlowp_uint8, uint8_t); \
  MACE_BM_MATMUL_FUNC(M, K, N, gemmlowp_int32, uint8_t);

// Embedding size 384
MACE_BM_MATMUL(7, 384, 384);
MACE_BM_MATMUL(7, 384, 1536);
MACE_BM_MATMUL(7, 1536, 384);

MACE_BM_MATMUL(15, 384, 384);
MACE_BM_MATMUL(15, 384, 1536);
MACE_BM_MATMUL(15, 1536, 384);

MACE_BM_MATMUL(1, 384, 384);
MACE_BM_MATMUL(1, 384, 1536);
MACE_BM_MATMUL(1, 1536, 384);
MACE_BM_MATMUL(1, 384, 44678);

// Embedding size 128
MACE_BM_MATMUL(1, 128, 1536);
MACE_BM_MATMUL(1, 128, 44678);

}  // namespace test
}  // namespace kernels
}  // namespace mace
