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

#include <algorithm>
#include <utility>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/utils/math.h"

#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/arm/fp32/gemv.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/arm/q8/gemv.h"
#endif  // MACE_ENABLE_QUANTIZE

#else
#include "mace/ops/ref/gemm.h"
#include "mace/ops/ref/gemv.h"
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/common/gemmlowp_util.h"
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/matmul.h"
#endif  // MACE_ENABLE_OPENCL
#ifdef MACE_ENABLE_NEON
#include "mace/ops/arm/fp16/gemv.h"
#endif

namespace mace {
namespace ops {

class MatMulOpBase : public Operation {
 public:
  explicit MatMulOpBase(OpConstructContext *context)
      : Operation(context),
        transpose_a_(Operation::GetOptionalArg<bool>("transpose_a", false)),
        transpose_b_(Operation::GetOptionalArg<bool>("transpose_b", false)) {}

  inline void Validate() {
    const Tensor *A = this->Input(INPUT_A);
    const Tensor *B = this->Input(INPUT_B);
    const index_t lhs_rank = A->dim_size();
    const index_t rhs_rank = B->dim_size();

    MACE_CHECK(lhs_rank >= 2 && rhs_rank >= 2,
               "rank should be greater than or equal to 2");
    if (lhs_rank == rhs_rank) {
      for (index_t i = 0; i < A->dim_size() - 2; ++i) {
        MACE_CHECK(A->dim(i) == B->dim(i),
                   "batch dimensions are not equal: ",
                   A->dim(i),
                   " vs. ",
                   B->dim(i));
      }
    } else {
      MACE_CHECK(lhs_rank == 2 || rhs_rank == 2,
                 "Either lhs or rhs matrix should has rank 2 "
                     "for non-batched matrix multiplication");
    }

    index_t
        lhs_depth = transpose_a_ ? A->dim(lhs_rank - 2) : A->dim(lhs_rank - 1);
    index_t
        rhs_depth = transpose_b_ ? B->dim(rhs_rank - 1) : B->dim(rhs_rank - 2);
    MACE_CHECK(lhs_depth == rhs_depth, "the number of A's column ", lhs_depth,
               " must be equal to B's row ", rhs_depth);
  }

 protected:
  MACE_OP_INPUT_TAGS(INPUT_A, INPUT_B, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

  bool transpose_a_;
  bool transpose_b_;
};

template<DeviceType D, class T>
class MatMulOp;

template<>
class MatMulOp<CPU, float> : public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    Validate();
    const Tensor *lhs = this->Input(INPUT_A);
    const Tensor *rhs = this->Input(INPUT_B);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *C = this->Output(OUTPUT);

    const index_t lhs_rank = lhs->dim_size();
    const index_t lhs_rows = lhs->dim(lhs_rank - 2);
    const index_t lhs_cols = lhs->dim(lhs_rank - 1);
    const index_t rhs_rank = rhs->dim_size();
    const index_t rhs_rows = rhs->dim(rhs_rank - 2);
    const index_t rhs_cols = rhs->dim(rhs_rank - 1);

    const index_t rows = transpose_a_ ? lhs_cols : lhs_rows;
    const index_t cols = transpose_b_ ? rhs_rows : rhs_cols;
    const index_t depth = transpose_a_ ? lhs_rows : lhs_cols;
    const index_t
        lhs_batch =
        std::accumulate(lhs->shape().begin(), lhs->shape().end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t
        rhs_batch =
        std::accumulate(rhs->shape().begin(), rhs->shape().end() - 2, 1,
                        std::multiplies<index_t>());
    index_t batch = 1;
    std::vector<index_t> output_shape;
    if (lhs_rank >= rhs_rank) {
      output_shape = lhs->shape();
      output_shape[lhs_rank - 2] = rows;
      output_shape[lhs_rank - 1] = cols;
      batch = lhs_batch;
    } else {
      output_shape = rhs->shape();
      output_shape[rhs_rank - 2] = rows;
      output_shape[rhs_rank - 1] = cols;
      batch = rhs_batch;
    }
    bool lhs_batched = true;
    bool rhs_batched = true;
    if (lhs_rank < rhs_rank) {
      lhs_batched = false;
    } else if (rhs_rank < lhs_rank) {
      rhs_batched = false;
    }

    MACE_RETURN_IF_ERROR(C->Resize(output_shape));

    if (rows == 1 && transpose_b_) {
      return gemv_.Compute(context,
                           rhs,
                           lhs,
                           bias,
                           batch,
                           cols,
                           depth,
                           rhs_batched,
                           lhs_batched,
                           C);
    } else if (cols == 1 && !transpose_a_) {
      return gemv_.Compute(context,
                           lhs,
                           rhs,
                           bias,
                           batch,
                           rows,
                           depth,
                           lhs_batched,
                           rhs_batched,
                           C);
    } else {
      context->device()->scratch_buffer()->Rewind();
      MaceStatus ret = gemm_.Compute(context,
                                     lhs,
                                     rhs,
                                     batch,
                                     lhs_rows,
                                     lhs_cols,
                                     rhs_rows,
                                     rhs_cols,
                                     transpose_a_,
                                     transpose_b_,
                                     false,
                                     lhs_batched,
                                     rhs_batched,
                                     C);
      if (bias != nullptr) {
        MACE_CHECK(bias->dim_size() == 1 && bias->dim(0) == cols,
                   "bias' dim should be <= 2.");
        Tensor::MappingGuard bias_guard(bias);
        Tensor::MappingGuard c_guard(C);
        const float *bias_data = bias->data<float>();
        float *c_data = C->mutable_data<float>();

        utils::ThreadPool
            &thread_pool = context->device()->cpu_runtime()->thread_pool();

        thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                  index_t start1, index_t end1, index_t step1) {
          for (index_t i = start0; i < end0; i += step0) {
            for (index_t w = start1; w < end1; w += step1) {
              c_data[i * cols + w] += bias_data[w];
            }
          }
        }, 0, batch * rows, 1, 0, cols, 1);
      }

      return ret;
    }
  }

 private:
#ifdef MACE_ENABLE_NEON
  arm::fp32::Gemm gemm_;
  arm::fp32::Gemv gemv_;
#else
  ref::Gemv<float> gemv_;
  ref::Gemm<float> gemm_;
#endif  // MACE_ENABLE_NEON
};

#ifdef MACE_ENABLE_QUANTIZE
template<gemmlowp::MapOrder AOrder, gemmlowp::MapOrder BOrder,
    typename OutputType>
class MatMulFixpointImpl;

template<gemmlowp::MapOrder AOrder, gemmlowp::MapOrder BOrder>
class MatMulFixpointImpl<AOrder, BOrder, uint8_t> {
 public:
  void operator()(OpContext *context,
                  const Tensor *A,
                  const Tensor *B,
                  const index_t batch,
                  const index_t height,
                  const index_t K,
                  const index_t width,
                  const bool lhs_batched,
                  const bool rhs_batched,
                  Tensor *C) {
#if defined(MACE_ENABLE_NEON)
    if (width == 1 && AOrder == gemmlowp::MapOrder::RowMajor) {
      gemv_kernel_.Compute(context,
                           A,
                           B,
                           nullptr,
                           batch,
                           height,
                           K,
                           lhs_batched,
                           rhs_batched,
                           C);
    } else if (height == 1 && BOrder == gemmlowp::MapOrder::ColMajor) {
      gemv_kernel_.Compute(context,
                           B,
                           A,
                           nullptr,
                           batch,
                           width,
                           K,
                           lhs_batched,
                           rhs_batched,
                           C);
    } else {
#endif  // MACE_ENABLE_NEON
      Tensor::MappingGuard guarda(A);
      Tensor::MappingGuard guardb(B);
      Tensor::MappingGuard guardc(C);
      auto a_ptr_base = A->data<uint8_t>();
      auto b_ptr_base = B->data<uint8_t>();
      auto c_ptr_base = C->mutable_data<uint8_t>();

      auto gemm_context =
          context->device()->cpu_runtime()->GetGemmlowpContext();
      MACE_CHECK_NOTNULL(gemm_context);

      index_t a_size = height * K;
      index_t b_size = K * width;
      index_t c_size = height * width;

      const auto &output_pipeline = GemmlowpOutputPipeline::MakeNoBias(
          A->scale(), B->scale(), C->scale(), C->zero_point());

      for (index_t i = 0; i < batch; ++i) {
        gemmlowp::MatrixMap<const uint8_t, AOrder>
            a_matrix(a_ptr_base
                         + static_cast<index_t>(lhs_batched) * i * a_size,
                     height,
                     K);
        gemmlowp::MatrixMap<const uint8_t, BOrder>
            b_matrix(b_ptr_base
                         + static_cast<index_t>(rhs_batched) * i * b_size,
                     K,
                     width);
        gemmlowp::MatrixMap <uint8_t, gemmlowp::MapOrder::RowMajor>
            c_matrix(c_ptr_base + i * c_size, height, width);

        using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
        gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
            gemm_context, a_matrix, b_matrix, &c_matrix, -A->zero_point(),
            -B->zero_point(), output_pipeline);
      }
    }
#if defined(MACE_ENABLE_NEON)
  }

 private:
  arm::q8::Gemv<uint8_t> gemv_kernel_;
#endif  // MACE_ENABLE_NEON
};

template<gemmlowp::MapOrder AOrder, gemmlowp::MapOrder BOrder>
class MatMulFixpointImpl<AOrder, BOrder, int32_t> {
 public:
  void operator()(OpContext *context,
                  const Tensor *A,
                  const Tensor *B,
                  const index_t batch,
                  const index_t height,
                  const index_t K,
                  const index_t width,
                  const bool lhs_batched,
                  const bool rhs_batched,
                  Tensor *C) {
    C->SetScale(A->scale() * B->scale());
    C->SetZeroPoint(0);

#if defined(MACE_ENABLE_NEON)
    if (width == 1 && AOrder == gemmlowp::MapOrder::RowMajor) {
      gemv_kernel_.Compute(context,
                           A,
                           B,
                           nullptr,
                           batch,
                           height,
                           K,
                           lhs_batched,
                           rhs_batched,
                           C);
    } else if (height == 1 && BOrder == gemmlowp::MapOrder::ColMajor) {
      gemv_kernel_.Compute(context,
                           B,
                           A,
                           nullptr,
                           batch,
                           width,
                           K,
                           lhs_batched,
                           rhs_batched,
                           C);
    } else {
#endif  // MACE_ENABLE_NEON
      Tensor::MappingGuard guarda(A);
      Tensor::MappingGuard guardb(B);
      Tensor::MappingGuard guardc(C);
      auto a_ptr_base = A->data<uint8_t>();
      auto b_ptr_base = B->data<uint8_t>();
      auto c_ptr_base = C->mutable_data<int32_t>();
      auto
          gemm_context =
          context->device()->cpu_runtime()->GetGemmlowpContext();
      MACE_CHECK_NOTNULL(gemm_context);

      index_t a_size = height * K;
      index_t b_size = K * width;
      index_t c_size = height * width;

      const auto output_pipeline = std::make_tuple();

      for (index_t i = 0; i < batch; ++i) {
        gemmlowp::MatrixMap<const uint8_t, AOrder>
            a_matrix
            (a_ptr_base + static_cast<index_t>(lhs_batched) * i * a_size,
             height,
             K);
        gemmlowp::MatrixMap<const uint8_t, BOrder>
            b_matrix
            (b_ptr_base + static_cast<index_t>(rhs_batched) * i * b_size,
             K,
             width);
        gemmlowp::MatrixMap <int32_t, gemmlowp::MapOrder::RowMajor>
            c_matrix(c_ptr_base + i * c_size, height, width);

        using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
        gemmlowp::GemmWithOutputPipeline<uint8_t, int32_t, BitDepthParams>(
            gemm_context, a_matrix, b_matrix, &c_matrix, -A->zero_point(),
            -B->zero_point(), output_pipeline);
      }
    }
#if defined(MACE_ENABLE_NEON)
  }

 private:
  arm::q8::Gemv<int32_t> gemv_kernel_;
#endif  // MACE_ENABLE_NEON
};

template<>
class MatMulOp<DeviceType::CPU, uint8_t> : public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    Validate();
    const Tensor *lhs = this->Input(INPUT_A);
    const Tensor *rhs = this->Input(INPUT_B);
    Tensor *C = this->Output(OUTPUT);

    const index_t lhs_rank = lhs->dim_size();
    const index_t lhs_rows = lhs->dim(lhs_rank - 2);
    const index_t lhs_cols = lhs->dim(lhs_rank - 1);
    const index_t rhs_rank = rhs->dim_size();
    const index_t rhs_rows = rhs->dim(rhs_rank - 2);
    const index_t rhs_cols = rhs->dim(rhs_rank - 1);

    const index_t rows = transpose_a_ ? lhs_cols : lhs_rows;
    const index_t cols = transpose_b_ ? rhs_rows : rhs_cols;
    const index_t depth = transpose_a_ ? lhs_rows : lhs_cols;
    const index_t
        lhs_batch =
        std::accumulate(lhs->shape().begin(), lhs->shape().end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t
        rhs_batch =
        std::accumulate(rhs->shape().begin(), rhs->shape().end() - 2, 1,
                        std::multiplies<index_t>());
    index_t batch = 1;
    std::vector<index_t> output_shape;
    if (lhs_rank >= rhs_rank) {
      output_shape = lhs->shape();
      output_shape[lhs_rank - 2] = rows;
      output_shape[lhs_rank - 1] = cols;
      batch = lhs_batch;
    } else {
      output_shape = rhs->shape();
      output_shape[rhs_rank - 2] = rows;
      output_shape[rhs_rank - 1] = cols;
      batch = rhs_batch;
    }
    bool lhs_batched = true;
    bool rhs_batched = true;
    if (lhs_rank < rhs_rank) {
      lhs_batched = false;
    } else if (rhs_rank < lhs_rank) {
      rhs_batched = false;
    }

    MACE_RETURN_IF_ERROR(C->Resize(output_shape));

    constexpr gemmlowp::MapOrder kRowMajor = gemmlowp::MapOrder::RowMajor;
    constexpr gemmlowp::MapOrder kColMajor = gemmlowp::MapOrder::ColMajor;

#define MATMUL_FIXPOINT_IMPL(AOrder, BOrder, OutType)           \
    MatMulFixpointImpl<AOrder, BOrder, OutType>()(              \
      context, lhs, rhs, batch, rows, depth, cols, lhs_batched, rhs_batched, C);

#define MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT(OutType)          \
    if (transpose_a_) {                                         \
      if (transpose_b_) {                                       \
        MATMUL_FIXPOINT_IMPL(kColMajor, kColMajor, OutType);    \
      } else {                                                  \
        MATMUL_FIXPOINT_IMPL(kColMajor, kRowMajor, OutType);    \
      }                                                         \
    } else {                                                    \
      if (transpose_b_) {                                       \
        MATMUL_FIXPOINT_IMPL(kRowMajor, kColMajor, OutType);    \
      } else {                                                  \
        MATMUL_FIXPOINT_IMPL(kRowMajor, kRowMajor, OutType);    \
      }                                                         \
    }

    if (!operator_def_->output_type().empty()
        && operator_def_->output_type()[0] == DT_INT32) {
      MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT(int32_t);
    } else {
      MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT(uint8_t);
    }

#undef MATMUL_FIXPOINT_IMPL_TRANSPOSE_OR_NOT
#undef MATMUL_FIXPOINT_IMPL

    return MaceStatus::MACE_SUCCESS;
  }
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class MatMulOp<DeviceType::GPU, T> : public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {
    MACE_UNUSED(context);
    MACE_NOT_IMPLEMENTED;
  }
  MaceStatus Run(OpContext *context) override {
    Validate();
    const Tensor *A = this->Input(INPUT_A);
    const Tensor *B = this->Input(INPUT_B);
    Tensor *C = this->Output(OUTPUT);
    return kernel_->Compute(context, A, B, C, transpose_a_, transpose_b_);
  }

 private:
  std::unique_ptr<OpenCLMatMulKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
template <>
class MatMulOp<CPU, float16_t> : public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_CHECK_NOTNULL(context);
    Validate();
    const Tensor *A = this->Input(INPUT_A);
    const Tensor *B = this->Input(INPUT_B);
    Tensor *C = this->Output(OUTPUT);

    index_t batch;
    index_t height;
    index_t K;
    index_t width;

    index_t rank = A->dim_size();
    height = A->dim(rank - 2);
    K = A->dim(rank - 1);
    if (transpose_a_) {
      std::swap(height, K);
    }
    if (transpose_b_) {
      width = B->dim(rank - 2);
    } else {
      width = B->dim(rank - 1);
    }
    batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                            std::multiplies<index_t>());

    std::vector<index_t> c_shape = A->shape();
    c_shape[rank - 2] = height;
    c_shape[rank - 1] = width;

    MACE_RETURN_IF_ERROR(C->Resize(c_shape));

    Tensor::MappingGuard guarda(A);
    Tensor::MappingGuard guardb(B);
    Tensor::MappingGuard guardc(C);
    auto *c_ptr_base = C->mutable_data<float>();

    MACE_CHECK(batch == 1, "matmul fp16 only support batch = 1 now");

    if (width == 1 && !transpose_a_ && A->dtype() == DT_FLOAT16 &&
        B->dtype() == DT_FLOAT) {
      auto *a_ptr_base = A->data<float16_t>();
      auto *b_ptr_base = B->data<float>();
      FP16Gemv(a_ptr_base,
               b_ptr_base,
               height,
               K,
               c_ptr_base);
      return MaceStatus::MACE_SUCCESS;
    } else if (height == 1 && transpose_b_ && A->dtype() == DT_FLOAT &&
               B->dtype() == DT_FLOAT16) {
      auto *b_ptr_base = B->data<float16_t>();
      auto *a_ptr_base = A->data<float>();
      FP16Gemv(b_ptr_base,
               a_ptr_base,
               width,
               K,
               c_ptr_base);
      return MaceStatus::MACE_SUCCESS;
    } else {
      LOG(INFO) << "Matmul fp16 gemv args: " << height << " " << width << " "
                << transpose_a_ << " " << transpose_b_;
      LOG(FATAL) << "Matmul fp16 Op only support fp32[1,k]·fp16[w,k]T or"
                    " fp16[w,k]·fp32[k,1] now!";
      return MaceStatus::MACE_INVALID_ARGS;
    }
  }

 private:
};
#endif  // MACE_ENABLE_NEON


void RegisterMatMul(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, float16_t);
#endif  // MACE_ENABLE_NEON
}

}  // namespace ops
}  // namespace mace
