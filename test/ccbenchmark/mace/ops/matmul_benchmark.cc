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

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "public/gemmlowp.h"
#include "mace/utils/statistics.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

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
namespace ops {
namespace test {

// Test the speed of different access order of a NHWC buffer

namespace {

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

#ifdef MACE_ENABLE_QUANTIZE
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

  auto gemm_context =
      mace::ops::test::OpTestContext::Get()
          ->GetDevice(CPU)->cpu_runtime()->GetGemmlowpContext();
  MACE_CHECK_NOTNULL(gemm_context);

  using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;

  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t, BitDepthParams>(
      gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
      -128, output_pipeline);

  mace::testing::StartTiming();
  while (iters--) {
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                     BitDepthParams>(
        gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
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

  auto gemm_context =
      mace::ops::test::OpTestContext::Get()
          ->GetDevice(CPU)->cpu_runtime()->GetGemmlowpContext();
  MACE_CHECK_NOTNULL(gemm_context);

  using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;

  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t, BitDepthParams>(
      gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
      -128, output_pipeline);

  mace::testing::StartTiming();
  while (iters--) {
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                     BitDepthParams>(
        gemm_context, lhs.const_map(), rhs.const_map(), &result.map(), -128,
        -128, output_pipeline);
  }
}
#endif

}  // namespace

#define MACE_BM_MATMUL_FUNC(M, K, N, FUNC, TYPE)                   \
  static void MACE_BM_MATMUL_##M##_##K##_##N##_##FUNC(int iters) { \
    const int64_t macs = static_cast<int64_t>(iters) *             \
        mace::benchmark::StatMACs("MatMul", {K}, {M, N});          \
    const int64_t tot = static_cast<int64_t>(iters) * (M + N) * K; \
    mace::testing::MacsProcessed(macs);                            \
    mace::testing::BytesProcessed(tot * sizeof(TYPE));             \
    MatmulBenchmark_##FUNC(iters, M, K, N);                        \
  }                                                                \
  MACE_BENCHMARK(MACE_BM_MATMUL_##M##_##K##_##N##_##FUNC)

#ifdef MACE_ENABLE_QUANTIZE
#define MACE_BM_MATMUL(M, K, N)                          \
  MACE_BM_MATMUL_FUNC(M, K, N, Eigen, float);            \
  MACE_BM_MATMUL_FUNC(M, K, N, gemmlowp_uint8, uint8_t); \
  MACE_BM_MATMUL_FUNC(M, K, N, gemmlowp_int32, uint8_t);
#else
#define MACE_BM_MATMUL(M, K, N)                          \
  MACE_BM_MATMUL_FUNC(M, K, N, Eigen, float)
#endif


// Embedding size 384
MACE_BM_MATMUL(7, 384, 384);
MACE_BM_MATMUL(7, 384, 1536);
MACE_BM_MATMUL(7, 1536, 384);

MACE_BM_MATMUL(15, 384, 384);
MACE_BM_MATMUL(15, 384, 1536);
MACE_BM_MATMUL(15, 1536, 384);

MACE_BM_MATMUL(1, 256, 256);
MACE_BM_MATMUL(1, 256, 1536);
MACE_BM_MATMUL(1, 1536, 256);
MACE_BM_MATMUL(256, 256, 1);
MACE_BM_MATMUL(1536, 256, 1);
MACE_BM_MATMUL(256, 1536, 1);
MACE_BM_MATMUL(29792, 256, 1);
MACE_BM_MATMUL(1, 256, 29792);
MACE_BM_MATMUL(2, 256, 256);
MACE_BM_MATMUL(2, 256, 1536);
MACE_BM_MATMUL(2, 1536, 256);
MACE_BM_MATMUL(3, 256, 256);
MACE_BM_MATMUL(3, 256, 1536);
MACE_BM_MATMUL(3, 1536, 256);
MACE_BM_MATMUL(4, 256, 256);
MACE_BM_MATMUL(4, 256, 1536);
MACE_BM_MATMUL(4, 1536, 256);
MACE_BM_MATMUL(8, 256, 256);
MACE_BM_MATMUL(8, 256, 1536);
MACE_BM_MATMUL(8, 1536, 256);
MACE_BM_MATMUL(10, 256, 256);
MACE_BM_MATMUL(10, 256, 1536);
MACE_BM_MATMUL(10, 1536, 256);
MACE_BM_MATMUL(15, 256, 256);
MACE_BM_MATMUL(15, 256, 1536);
MACE_BM_MATMUL(15, 1536, 256);

// Embedding size 128
MACE_BM_MATMUL(1, 128, 1536);
MACE_BM_MATMUL(1, 128, 44678);

// MobileNet
MACE_BM_MATMUL(128, 128, 3136);
MACE_BM_MATMUL(256, 256, 784);
MACE_BM_MATMUL(512, 512, 196);
MACE_BM_MATMUL(1024, 1024, 49);

namespace {
template<DeviceType D, typename T>
void MatMulBenchmark(
    int iters, int batch, int height, int channels, int out_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  if (DataTypeToEnum<T>::value == DT_FLOAT16) {
    net.AddRandomInput<D, float16_t>("A", {batch, height, channels});
    net.AddRandomInput<D, float>("B", {batch, channels, out_width});
  } else {
#endif
    net.AddRandomInput<D, T>("A", {batch, height, channels});
    net.AddRandomInput<D, T>("B", {batch, channels, out_width});
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  }
#endif
  net.GetTensor("A")->SetIsWeight(true);
  net.GetTensor("B")->SetIsWeight(true);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("A")->SetScale(0.1);
    net.GetTensor("B")->SetScale(0.1);
  }
  OpDefBuilder("MatMul", "MatMulBM")
      .Input("A")
      .Input("B")
      .Output("Output")
      .OutputType({DT_INT32})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  net.Setup(D);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Output")->SetScale(0.1);
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}

template<DeviceType D, typename T>
void MatMulTransposeBenchmark(
    int iters, int batch, int height, int channels, int out_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  if (DataTypeToEnum<T>::value == DT_FLOAT16) {
    net.AddRandomInput<D, float>("A", {batch, height, channels});
    net.AddRandomInput<D, float16_t>("B", {batch, out_width, channels});
  } else {
#endif
    net.AddRandomInput<D, T>("A", {batch, height, channels});
    net.AddRandomInput<D, float>("B", {batch, out_width, channels});
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  }
#endif
  net.GetTensor("A")->SetIsWeight(true);
  net.GetTensor("B")->SetIsWeight(true);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("A")->SetScale(0.1);
    net.GetTensor("B")->SetScale(0.1);
  }

  if (D == DeviceType::CPU) {
    OpDefBuilder("MatMul", "MatMulBM")
        .Input("A")
        .Input("B")
        .AddIntArg("transpose_b", 1)
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  net.Setup(D);
  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Output")->SetScale(0.1);
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_MATMUL_MACRO(N, H, C, W, TYPE, DEVICE)                         \
  static void MACE_BM_MATMUL_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE(        \
      int iters) {                                                             \
    const int64_t macs = static_cast<int64_t>(iters) *                         \
        mace::benchmark::StatMACs("MatMul", {C}, {N, H, W});                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * (C * H + H * W);     \
    mace::testing::MacsProcessed(macs);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    MatMulBenchmark<DEVICE, TYPE>(iters, N, H, C, W);                          \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_MATMUL_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_QUANTIZE
#define MACE_BM_MATMUL_OP(N, H, C, W)              \
  MACE_BM_MATMUL_MACRO(N, H, C, W, float, CPU);    \
  MACE_BM_MATMUL_MACRO(N, H, C, W, uint8_t, CPU)
#else
#define MACE_BM_MATMUL_OP(N, H, C, W)              \
  MACE_BM_MATMUL_MACRO(N, H, C, W, float, CPU)
#endif

#define MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, TYPE, DEVICE)               \
  static void MACE_BM_MATMUL_##T_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE(    \
      int iters) {                                                             \
    const int64_t macs = static_cast<int64_t>(iters) *                         \
        mace::benchmark::StatMACs("MatMul", {C}, {N, H, W});                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * (C * H + H * W);     \
    mace::testing::MacsProcessed(macs);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    MatMulTransposeBenchmark<DEVICE, TYPE>(iters, N, H, C, W);                 \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_MATMUL_##T_##N##_##H##_##C##_##W##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_QUANTIZE
#define MACE_BM_MATMUL_TRANPOSE(N, H, C, W)                   \
  MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, float, CPU);     \
  MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, uint8_t, CPU);
#else
#define MACE_BM_MATMUL_TRANPOSE(N, H, C, W)                   \
  MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, float, CPU);
#endif

MACE_BM_MATMUL_OP(1, 30000, 256, 1);
MACE_BM_MATMUL_OP(1, 128, 256, 128);
MACE_BM_MATMUL_OP(2, 128, 128, 49);
MACE_BM_MATMUL_OP(3, 128, 128, 49);
MACE_BM_MATMUL_OP(4, 128, 128, 49);
MACE_BM_MATMUL_OP(16, 32, 128, 49);
MACE_BM_MATMUL_OP(16, 32, 128, 961);
MACE_BM_MATMUL_OP(16, 32, 128, 3969);
MACE_BM_MATMUL_OP(16, 128, 128, 49);
MACE_BM_MATMUL_OP(16, 49, 128, 128);
MACE_BM_MATMUL_OP(16, 128, 128, 961);
MACE_BM_MATMUL_OP(16, 128, 128, 3969);

MACE_BM_MATMUL_TRANPOSE(16, 32, 128, 49);
MACE_BM_MATMUL_TRANPOSE(16, 32, 128, 961);
MACE_BM_MATMUL_TRANPOSE(16, 32, 128, 3969);
MACE_BM_MATMUL_TRANPOSE(16, 128, 128, 49);
MACE_BM_MATMUL_TRANPOSE(16, 128, 128, 961);
MACE_BM_MATMUL_TRANPOSE(16, 128, 128, 3969);

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
#define MACE_BM_MATMUL_TRANPOSE_FP16(N, H, C, W)              \
  MACE_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, float16_t, CPU);

MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 256, 30000);
MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 256, 256);
MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 256, 2048);
MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 2048, 256);

MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 512, 30000);
MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 512, 512);
MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 512, 2048);
MACE_BM_MATMUL_TRANPOSE_FP16(1, 1, 2048, 512);
#endif  // MACE_ENABLE_NEON

}  // namespace test
}  // namespace ops
}  // namespace mace
