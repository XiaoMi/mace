// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/activation.h"
#include "mace/ops/common/eltwise_type.h"
#include "mace/ops/common/reduce_type.h"
#include "mace/ops/delegator/activation.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/instance_norm.h"
#include "mace/ops/opencl/image/eltwise.h"
#include "mace/ops/opencl/image/reduce.h"
#include "mace/runtimes/opencl/transform/buffer_transformer.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<RuntimeType D, class T>
class InstanceNormOp;

template<class T>
class InstanceNormOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit InstanceNormOp(OpConstructContext *context)
      : Operation(context),
        epsilon_(Operation::GetOptionalArg<float>("epsilon",
                                                  static_cast<float>(1e-4))),
        affine_(Operation::GetOptionalArg<bool>("affine", false)),
        activation_delegator_(
            delegator::Activation::Create(
                context->workspace(),
                MACE_DELEGATOR_KEY(Activation, RuntimeType::RT_CPU,
                                   T, kCpuImplType),
                delegator::ActivationParam(
                    ops::StringToActivationType(
                        Operation::GetOptionalArg<std::string>("activation",
                                                               "NOOP")),
                    Operation::GetOptionalArg<float>("max_limit", 0.0f),
                    Operation::GetOptionalArg<float>("activation_coefficient",
                                                     0.0f),
                    Operation::GetOptionalArg<float>("hardsigmoid_alpha", 0.f),
                    Operation::GetOptionalArg<float>("hardsigmoid_beta", 0.f)))) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *scale = nullptr;
    const Tensor *offset = nullptr;
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    if (affine_) {
      scale = this->Input(SCALE);
      offset = this->Input(OFFSET);
    }
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());
    if (affine_) {
      MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional.",
                 scale->dim_size());
      MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional.",
                 offset->dim_size());
    }


    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);
    index_t height_width = height * width;
    index_t batch_channel = batch * channels;

    utils::ThreadPool &thread_pool = context->runtime()->thread_pool();
    mean_.reserve(batch_channel);
    T* mean_ptr = mean_.data();
    var_.reserve(batch_channel);
    T* var_ptr = var_.data();
    T* diff_square_ptr = output_ptr;

    thread_pool.Compute1D([=](index_t start,
                              index_t end,
                              index_t step) {
      for (index_t i = start; i < end; i += step) {
        T accumulation = 0.f;
        index_t base_idx = i * height_width;
        for (index_t j = 0; j < height_width; ++j) {
          accumulation += input_ptr[base_idx + j];
        }
        mean_ptr[i] = accumulation / height_width;
        T diff = 0.f;
        for (index_t j = 0; j < height_width; ++j) {
          index_t idx = base_idx + j;
          diff = input_ptr[idx] - mean_ptr[i];
          diff_square_ptr[idx] =  diff * diff;
        }
      }
    }, 0, batch_channel, 1);

    thread_pool.Compute1D([=](index_t start,
                              index_t end,
                              index_t step) {
      for (index_t i = start; i < end; i += step) {
        T accumulation = 0.f;
        index_t base_idx = i * height_width;
        for (index_t j = 0; j < height_width; ++j) {
          accumulation += diff_square_ptr[base_idx + j];
        }
        var_ptr[i] = accumulation / height_width;
      }
    }, 0, batch_channel, 1);

    if (affine_) {
      const T *scale_ptr = scale->data<T>();
      const T *offset_ptr = offset->data<T>();
      // If current capacity is big enough, reserve does nothing
      new_scale_.reserve(batch_channel);
      T* new_scale_ptr = new_scale_.data();
      new_offset_.reserve(batch_channel);
      T* new_offset_ptr = new_offset_.data();
      thread_pool.Compute1D([=](index_t start,
                                index_t end,
                                index_t step) {
        for (index_t i = start; i < end; i += step) {
          index_t base_idx = i  * channels;
          for (index_t j = 0; j < channels; ++j) {
            index_t idx = base_idx + j;
            new_scale_ptr[idx] =
                scale_ptr[j]  / std::sqrt(var_ptr[idx] + epsilon_);
            new_offset_ptr[idx] =
                offset_ptr[j] - mean_ptr[idx] * new_scale_ptr[idx];
          }
        }
      }, 0, batch, 1);

      thread_pool.Compute1D([=](index_t start,
                                index_t end,
                                index_t step) {
        for (index_t i = start; i < end; i += step) {
          index_t base_idx = i * height_width;
          for (index_t j = 0; j < height_width; ++j) {
            index_t idx = base_idx + j;
            output_ptr[idx] =
                input_ptr[idx] * new_scale_ptr[i] + new_offset_ptr[i];
          }
        }
      }, 0, batch_channel, 1);
    } else {
      new_scale_.reserve(batch_channel);
      T* new_scale_ptr = new_scale_.data();
      thread_pool.Compute1D([=](index_t start,
                                index_t end,
                                index_t step) {
        for (index_t i = start; i < end; i += step) {
          new_scale_ptr[i] = 1. / std::sqrt(var_ptr[i] + epsilon_);
        }
      }, 0, batch_channel, 1);
      thread_pool.Compute1D([=](index_t start,
                                index_t end,
                                index_t step) {
        for (index_t i = start; i < end; i += step) {
          index_t base_idx = i * height_width;
          for (index_t j = 0; j < height_width; ++j) {
            index_t idx = base_idx + j;
            output_ptr[idx] =
                (input_ptr[idx] - mean_ptr[i]) * new_scale_ptr[i];
          }
        }
      }, 0, batch_channel, 1);
    }

    activation_delegator_->Compute(context, output, output);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  float epsilon_;
  bool affine_;
  std::vector<T> mean_;
  std::vector<T> var_;
  std::vector<T> new_scale_;
  std::vector<T> new_offset_;
  std::unique_ptr<delegator::Activation> activation_delegator_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, SCALE, OFFSET);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};


#ifdef MACE_ENABLE_OPENCL
template<>
class InstanceNormOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit InstanceNormOp(OpConstructContext *context)
      : Operation(context) {
    const float epsilon = Operation::GetOptionalArg<float>(
        "epsilon", static_cast<float>(1e-4));
    ActivationType activation = ops::StringToActivationType(
        Operation::GetOptionalArg<std::string>("activation", "NOOP"));
    const float relux_max_limit = Operation::GetOptionalArg<float>(
        "max_limit", 0.0f);
    const float activation_coefficient = Operation::GetOptionalArg<float>(
        "activation_coefficient", 0.0f);
    affine_ = Operation::GetOptionalArg<bool>("affine", false);

    Runtime *runtime = context->runtime();
    mace::DataType dt = static_cast<mace::DataType>(
        Operation::GetOptionalArg<int>(
        "T", static_cast<int>(mace::DataType::DT_FLOAT)));
    mean_tensor_ = make_unique<Tensor>(
        runtime, dt, MemoryType::GPU_IMAGE,
        std::vector<index_t>(), false, "mean_tensor_");
    var_tensor_ = make_unique<Tensor>(
        runtime, dt, MemoryType::GPU_IMAGE,
        std::vector<index_t>(), false, "var_tensor_");
    MemoryType mem_type;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      std::vector<int> axis = {1, 2};
      reduce_kernel_ = make_unique<opencl::image::ReduceKernel>(
          ReduceType::MEAN, axis);
      std::vector<float> coeff;
      eltwise_kernel_ = make_unique<opencl::image::EltwiseKernel>(
          EltwiseType::SQR_DIFF, coeff, 1., 1);
      mem_type = MemoryType::GPU_IMAGE;
      instance_norm_kernel_ = make_unique<opencl::image::InstanceNormKernel>(
          epsilon, activation, relux_max_limit, activation_coefficient,
          affine_);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    if (affine_) {
      // Transform filters
      int input_size = operator_def_->input_size();
      for (int i = 1; i < input_size; ++i) {
        const Tensor *input_tensor = context->workspace()->GetTensor(
            operator_def_->input(i));
        MACE_CHECK(input_tensor != nullptr);
        MACE_CHECK(TransformFilter(
            context,
            operator_def_.get(),
            i,
            BufferContentType::ARGUMENT,
            mem_type) == MaceStatus::MACE_SUCCESS);
      }
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(3);
    MACE_RETURN_IF_ERROR(mean_tensor_->Resize({batch, 1, 1, channels}));
    MACE_RETURN_IF_ERROR(var_tensor_->Resize({batch, 1, 1, channels}));
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    reduce_kernel_->Compute(context, input, mean_tensor_.get());
    // Reuse output to hold (X-E)^2 as intermediate GPU memory.
    eltwise_kernel_->Compute(context, input, mean_tensor_.get(),
                             output);
    reduce_kernel_->Compute(context, output, var_tensor_.get());
    const Tensor *scale = nullptr;
    const Tensor *offset = nullptr;
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    if (affine_) {
      scale = this->Input(SCALE);
      offset = this->Input(OFFSET);
      MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
                 scale->dim_size());
      MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
                 offset->dim_size());
    }

    MaceStatus status = instance_norm_kernel_->Compute(
        context, input, scale, offset,
        mean_tensor_.get(), var_tensor_.get(), output);
    if (status != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "InstanceNorm kernel computation failed";
    }
    return status;
  }

 private:
  std::unique_ptr<OpenCLReduceKernel> reduce_kernel_;
  std::unique_ptr<OpenCLEltwiseKernel> eltwise_kernel_;
  std::unique_ptr<OpenCLInstanceNormKernel> instance_norm_kernel_;

  std::unique_ptr<Tensor> mean_tensor_;
  std::unique_ptr<Tensor> var_tensor_;
  bool affine_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, SCALE, OFFSET);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif


void RegisterInstanceNorm(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "InstanceNorm", InstanceNormOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "InstanceNorm",
                        InstanceNormOp, RuntimeType::RT_CPU);
  MACE_REGISTER_FP16_OP(op_registry, "InstanceNorm",
                        InstanceNormOp, RuntimeType::RT_CPU);
  MACE_REGISTER_GPU_OP(op_registry, "InstanceNorm", InstanceNormOp);
}

}  // namespace ops
}  // namespace mace
