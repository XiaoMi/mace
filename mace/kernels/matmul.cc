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
#include <utility>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/kernels/gemm.h"
#include "mace/kernels/gemmlowp_util.h"
#include "mace/kernels/sgemm.h"
#include "mace/utils/utils.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/kernels/opencl/image/matmul.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

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
class MatMulOp : public MatMulOpBase {
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
    const T *a_ptr_base = A->data<T>();
    const T *b_ptr_base = B->data<T>();
    T *c_ptr_base = C->mutable_data<T>();

    const index_t height_a = A->dim(rank - 2);
    const index_t width_a = A->dim(rank - 1);
    const index_t height_b = B->dim(rank - 2);
    const index_t width_b = B->dim(rank - 1);

    auto scratch_buffer = context->device()->scratch_buffer();
    scratch_buffer->Rewind();
    index_t scratch_size = C->raw_max_size();
    if (!A->is_weight()) {
      scratch_size += A->raw_max_size();
    }
    if (!B->is_weight()) {
      scratch_size += B->raw_max_size();
    }
    scratch_buffer->GrowSize(scratch_size);

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

#define MATMUL_IMPL(AOrder, BOrder) \
    MatMulImpl<AOrder, BOrder>(context, A, B, height, K, width, C);

    if (transpose_a_) {
      if (transpose_b_) {
        MATMUL_IMPL(kColMajor, kColMajor);
      } else {
        MATMUL_IMPL(kColMajor, kRowMajor);
      }
    } else {
      if (transpose_b_) {
        MATMUL_IMPL(kRowMajor, kColMajor);
      } else {
        MATMUL_IMPL(kRowMajor, kRowMajor);
      }
    }

#undef MATMUL_IMPL

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  template<gemmlowp::MapOrder AOrder, gemmlowp::MapOrder BOrder>
  void MatMulImpl(OpContext *context,
                  const Tensor *A,
                  const Tensor *B,
                  const index_t height,
                  const index_t K,
                  const index_t width,
                  Tensor *C) {
    auto gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
    MACE_CHECK_NOTNULL(gemm_context);

    Tensor::MappingGuard guarda(A);
    Tensor::MappingGuard guardb(B);
    Tensor::MappingGuard guardc(C);
    auto a_ptr_base = A->data<uint8_t>();
    auto b_ptr_base = B->data<uint8_t>();
    auto c_ptr_base = C->mutable_data<uint8_t>();
    index_t batch = std::accumulate(A->shape().begin(), A->shape().end() - 2, 1,
                                    std::multiplies<index_t>());
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
      gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::RowMajor>
          c_matrix(c_ptr_base + i * c_size, height, width);

      using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
      gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
          gemm_context, a_matrix, b_matrix, &c_matrix, -A->zero_point(),
          -B->zero_point(), output_pipeline);
    }
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class MatMulOp<DeviceType::GPU, T> : public MatMulOpBase {
 public:
  explicit MatMulOp(OpConstructContext *context)
      : MatMulOpBase(context) {
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::MatMulKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
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

  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::CPU, uint8_t);
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "MatMul", MatMulOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace kernels
}  // namespace mace
