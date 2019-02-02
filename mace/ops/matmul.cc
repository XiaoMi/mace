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
#include "mace/ops/sgemm.h"
#include "mace/utils/utils.h"

#ifdef MACE_ENABLE_NEON

#include "mace/ops/arm/fp32/gemv.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/arm/q8/gemv.h"
#endif  // MACE_ENABLE_QUANTIZE

#else
#include "mace/ops/ref/gemv.h"
#endif  // MACE_ENABLE_NEON

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/gemmlowp_util.h"
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/matmul.h"
#endif  // MACE_ENABLE_OPENCL

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
    MACE_CHECK(A->dim_size() == B->dim_size() && A->dim_size() >= 2,
               "rank(A) should be equal to rank(B), rank should be greater "
               "than or equal to 2");
    index_t rank = A->dim_size();
    for (index_t i = 0; i < rank - 2; ++i) {
      MACE_CHECK(A->dim(i) == B->dim(i),
                 "batch dimensions are not equal: ",
                 A->dim(i),
                 " vs. ",
                 B->dim(i));
    }
    index_t ak = transpose_a_ ? A->dim(rank - 2) : A->dim(rank - 1);
    index_t bk = transpose_b_ ? B->dim(rank - 1) : B->dim(rank - 2);
    MACE_CHECK(ak == bk, "the number of A's column ", ak,
               " must be equal to B's row ", bk);
  }

 protected:
  MACE_OP_INPUT_TAGS(INPUT_A, INPUT_B);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

  bool transpose_a_;
  bool transpose_b_;
};

template <DeviceType D, class T>
class MatMulOp;

template <>
class MatMulOp<CPU, float> : public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
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
    const float *a_ptr_base = A->data<float>();
    const float *b_ptr_base = B->data<float>();
    float *c_ptr_base = C->mutable_data<float>();

    const index_t height_a = A->dim(rank - 2);
    const index_t width_a = A->dim(rank - 1);
    const index_t height_b = B->dim(rank - 2);
    const index_t width_b = B->dim(rank - 1);

    auto scratch_buffer = context->device()->scratch_buffer();
    scratch_buffer->Rewind();

    sgemm_.Run(a_ptr_base,
               b_ptr_base,
               batch,
               height_a,
               width_a,
               height_b,
               width_b,
               transpose_a_,
               transpose_b_,
               A->is_weight(),
               B->is_weight(),
               c_ptr_base,
               context->device()->scratch_buffer());

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  SGemm sgemm_;
#ifdef MACE_ENABLE_NEON
  arm::fp32::Gemv gemv_;
#else
  ref::Gemv<float> gemv_;
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
                  const index_t height,
                  const index_t K,
                  const index_t width,
                  Tensor *C) {
    index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                    std::multiplies<index_t>());

#if defined(MACE_ENABLE_NEON)
    if (width == 1 && AOrder == gemmlowp::MapOrder::RowMajor) {
      gemv_kernel_.Compute(context, A, B, nullptr, batch, height, K, true, C);
    } else if (height == 1 && BOrder == gemmlowp::MapOrder::ColMajor) {
      gemv_kernel_.Compute(context, B, A, nullptr, batch, width, K, true, C);
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
            a_matrix(a_ptr_base + i * a_size, height, K);
        gemmlowp::MatrixMap<const uint8_t, BOrder>
            b_matrix(b_ptr_base + i * b_size, K, width);
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
                  const index_t height,
                  const index_t K,
                  const index_t width,
                  Tensor *C) {
    C->SetScale(A->scale() * B->scale());
    C->SetZeroPoint(0);
    index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                    std::multiplies<index_t>());

#if defined(MACE_ENABLE_NEON)
    if (width == 1 && AOrder == gemmlowp::MapOrder::RowMajor) {
      gemv_kernel_.Compute(context, A, B, nullptr, batch, height, K, true, C);
    } else if (height == 1 && BOrder == gemmlowp::MapOrder::ColMajor) {
      gemv_kernel_.Compute(context, B, A, nullptr, batch, width, K, true, C);
    } else {
#endif  // MACE_ENABLE_NEON
      Tensor::MappingGuard guarda(A);
      Tensor::MappingGuard guardb(B);
      Tensor::MappingGuard guardc(C);
      auto a_ptr_base = A->data<uint8_t>();
      auto b_ptr_base = B->data<uint8_t>();
      auto c_ptr_base = C->mutable_data<int32_t>();
      auto
          gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
      MACE_CHECK_NOTNULL(gemm_context);

      index_t a_size = height * K;
      index_t b_size = K * width;
      index_t c_size = height * width;

      const auto output_pipeline = std::make_tuple();

      for (index_t i = 0; i < batch; ++i) {
        gemmlowp::MatrixMap<const uint8_t, AOrder>
            a_matrix(a_ptr_base + i * a_size, height, K);
        gemmlowp::MatrixMap<const uint8_t, BOrder>
            b_matrix(b_ptr_base + i * b_size, K, width);
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

template <>
class MatMulOp<DeviceType::CPU, uint8_t>: public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    Validate();
    const Tensor *A = this->Input(INPUT_A);
    const Tensor *B = this->Input(INPUT_B);
    Tensor *C = this->Output(OUTPUT);

    index_t rank = A->dim_size();
    index_t height = A->dim(rank - 2);
    index_t K = A->dim(rank - 1);
    index_t width;

    if (transpose_a_) {
      std::swap(height, K);
    }
    if (transpose_b_) {
      width = B->dim(rank - 2);
    } else {
      width = B->dim(rank - 1);
    }

    std::vector<index_t> c_shape = A->shape();
    c_shape[rank - 2] = height;
    c_shape[rank - 1] = width;

    MACE_RETURN_IF_ERROR(C->Resize(c_shape));

    constexpr gemmlowp::MapOrder kRowMajor = gemmlowp::MapOrder::RowMajor;
    constexpr gemmlowp::MapOrder kColMajor = gemmlowp::MapOrder::ColMajor;

#define MATMUL_FIXPOINT_IMPL(AOrder, BOrder, OutType)           \
    MatMulFixpointImpl<AOrder, BOrder, OutType>()(              \
      context, A, B, height, K, width, C);

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
}

}  // namespace ops
}  // namespace mace
