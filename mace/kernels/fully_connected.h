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

#ifndef MACE_KERNELS_FULLY_CONNECTED_H_
#define MACE_KERNELS_FULLY_CONNECTED_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/gemm.h"
#include "mace/kernels/gemmlowp_util.h"

namespace mace {
namespace kernels {

struct FullyConnectedBase {
  FullyConnectedBase(const ActivationType activation,
                     const float relux_max_limit)
      : activation_(activation),
        relux_max_limit_(relux_max_limit) {}

  const ActivationType activation_;
  const float relux_max_limit_;
};

template <DeviceType D, typename T>
struct FullyConnectedFunctor;

template <>
struct FullyConnectedFunctor<DeviceType::CPU, float>: FullyConnectedBase {
  FullyConnectedFunctor(const ActivationType activation,
                        const float relux_max_limit)
      : FullyConnectedBase(activation, relux_max_limit) {}

  MaceStatus operator()(const Tensor *input,
                        const Tensor *weight,
                        const Tensor *bias,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    std::vector<index_t> output_shape = {input->dim(0), weight->dim(0), 1, 1};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const index_t N = output->dim(0);
    const index_t input_size = weight->dim(1) * weight->dim(2) * weight->dim(3);
    const index_t output_size = weight->dim(0);

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_weight(weight);
    Tensor::MappingGuard guard_output(output);
    const float *input_ptr = input->data<float>();
    const float *weight_ptr = weight->data<float>();
    float *output_ptr = output->mutable_data<float>();

    Gemv(weight_ptr, input_ptr, N, input_size, output_size, output_ptr);

    if (bias) {
      Tensor::MappingGuard guard_bias(bias);
      const float *bias_ptr = bias == nullptr ? nullptr : bias->data<float>();
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < output_size; ++j) {
          output_ptr[j + i * output_size] += bias_ptr[j];
        }
      }
    }

    DoActivation(output_ptr, output_ptr, output->size(), activation_,
                 relux_max_limit_);

    return MACE_SUCCESS;
  }
};

template <>
struct FullyConnectedFunctor<DeviceType::CPU, uint8_t>: FullyConnectedBase {
  FullyConnectedFunctor(const ActivationType activation,
                        const float relux_max_limit)
      : FullyConnectedBase(activation, relux_max_limit) {}

  MaceStatus operator()(const Tensor *input,
                        const Tensor *weight,
                        const Tensor *bias,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    gemmlowp::GemmContext& gemm_context = GetGemmlowpContext();

    std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    const int N = static_cast<int>(output->dim(0));
    const int input_size =
        static_cast<int>(weight->dim(1) * weight->dim(2) * weight->dim(3));
    const int output_size = static_cast<int>(weight->dim(0));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_weight(weight);
    Tensor::MappingGuard guard_output(output);
    auto input_ptr = input->data<uint8_t>();
    auto weight_ptr = weight->data<uint8_t>();
    auto output_ptr = output->mutable_data<uint8_t>();

    std::vector<index_t> bias_shape{output_size};
    std::unique_ptr<Tensor> zero_bias;
    const int32_t *bias_ptr = nullptr;
    if (bias == nullptr) {
      zero_bias.reset(
          new Tensor(GetDeviceAllocator(DeviceType::CPU), DT_INT32));
      zero_bias->Resize(bias_shape);
      zero_bias->Clear();
      bias_ptr = zero_bias->data<int32_t>();
    } else {
      bias_ptr = bias->data<int32_t>();
    }

    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
        weight_matrix(weight_ptr, output_size, input_size);
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
        input_matrix(input_ptr, input_size, N);
    gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::ColMajor>
        output_matrix(output_ptr, output_size, N);

    const auto &output_pipeline = GemmlowpOutputPipeline::Make(
        bias_ptr, output_size, weight->scale(), input->scale(), output->scale(),
        output->zero_point());

    using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
    gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
        &gemm_context, weight_matrix, input_matrix, &output_matrix,
        -weight->zero_point(), -input->zero_point(), output_pipeline);

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct FullyConnectedFunctor<DeviceType::GPU, T> : FullyConnectedBase {
  FullyConnectedFunctor(const ActivationType activation,
                        const float relux_max_limit)
      : FullyConnectedBase(activation, relux_max_limit) {}

  MaceStatus operator()(const Tensor *input,
                  const Tensor *weight,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  std::vector<uint32_t> gws_;
  std::vector<uint32_t> lws_;
  std::vector<index_t> input_shape_;
  std::unique_ptr<BufferBase> kernel_error_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_FULLY_CONNECTED_H_
