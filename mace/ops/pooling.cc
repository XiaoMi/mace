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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include "mace/ops/pooling.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/pooling.h"
#include "mace/ops/opencl/buffer/pooling.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

class PoolingOpBase : public ConvPool2dOpBase {
 public:
  explicit PoolingOpBase(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        kernels_(Operation::GetRepeatedArgs<int>("kernels")),
        pooling_type_(
            static_cast<PoolingType>(Operation::GetOptionalArg<int>(
                "pooling_type", static_cast<int>(AVG)))),
        round_type_(static_cast<RoundType>(Operation::GetOptionalArg<int>(
            "round_mode", static_cast<int>(CEIL)))) {}

 protected:
  std::vector<int> kernels_;
  PoolingType pooling_type_;
  RoundType round_type_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

template<DeviceType D, class T>
class PoolingOp;

template<>
class PoolingOp<DeviceType::CPU, float> : public PoolingOpBase {
 public:
  explicit PoolingOp(OpConstructContext *context)
      : PoolingOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input_tensor = this->Input(0);
    Tensor *output_tensor = this->Output(0);
    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {
        input_tensor->dim(1), input_tensor->dim(1), kernels_[0], kernels_[1]};

    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      ops::CalcNCHWPaddingAndOutputSize(
          input_tensor->shape().data(), filter_shape.data(), dilations_.data(),
          strides_.data(), padding_type_, output_shape.data(), paddings.data());
    } else {
      paddings = paddings_;
      CalcNCHWOutputSize(input_tensor->shape().data(),
                         filter_shape.data(),
                         paddings_.data(),
                         dilations_.data(),
                         strides_.data(),
                         round_type_,
                         output_shape.data());
    }
    MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

    Tensor::MappingGuard input_guard(input_tensor);
    Tensor::MappingGuard output_guard(output_tensor);
    const float *input = input_tensor->data<float>();
    float *output = output_tensor->mutable_data<float>();
    const index_t *input_shape = input_tensor->shape().data();
    int pad_hw[2] = {paddings[0] / 2, paddings[1] / 2};

    if (pooling_type_ == PoolingType::MAX) {
      MaxPooling(context,
                 input,
                 input_shape,
                 output_shape.data(),
                 kernels_.data(),
                 strides_.data(),
                 dilations_.data(),
                 pad_hw,
                 output);
    } else if (pooling_type_ == PoolingType::AVG) {
      AvgPooling(context,
                 input,
                 input_shape,
                 output_shape.data(),
                 kernels_.data(),
                 strides_.data(),
                 dilations_.data(),
                 pad_hw,
                 output);
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void MaxPooling(const OpContext *context,
                  const float *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *dilation_hw,
                  const int *pad_hw,
                  float *output) {
    const index_t batch = out_shape[0];
    const index_t out_channels = out_shape[1];
    const index_t out_height = out_shape[2];
    const index_t out_width = out_shape[3];
    const index_t in_channels = in_shape[1];
    const index_t in_height = in_shape[2];
    const index_t in_width = in_shape[3];

    const index_t in_image_size = in_height * in_width;
    const index_t out_image_size = out_height * out_width;
    const index_t in_batch_size = in_channels * in_image_size;
    const index_t out_batch_size = out_channels * out_image_size;

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t c = start1; c < end1; c += step1) {
          const index_t out_base = b * out_batch_size + c * out_image_size;
          const index_t in_base = b * in_batch_size + c * in_image_size;

          for (index_t h = 0; h < out_height; ++h) {
            for (index_t w = 0; w < out_width; ++w) {
              const index_t out_offset = out_base + h * out_width + w;
              float res = std::numeric_limits<float>::lowest();
              for (int fh = 0; fh < filter_hw[0]; ++fh) {
                for (int fw = 0; fw < filter_hw[1]; ++fw) {
                  index_t inh =
                      h * stride_hw[0] + dilation_hw[0] * fh - pad_hw[0];
                  index_t inw =
                      w * stride_hw[1] + dilation_hw[1] * fw - pad_hw[1];
                  if (inh >= 0 && inh < in_height && inw >= 0
                      && inw < in_width) {
                    index_t input_offset = in_base + inh * in_width + inw;
                    res = std::max(res, input[input_offset]);
                  }
                }
              }
              output[out_offset] = res;
            }
          }
        }
      }
    }, 0, batch, 1, 0, out_channels, 1);
  }

  void AvgPooling(const OpContext *context,
                  const float *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *dilation_hw,
                  const int *pad_hw,
                  float *output) {
    const index_t batch = out_shape[0];
    const index_t out_channels = out_shape[1];
    const index_t out_height = out_shape[2];
    const index_t out_width = out_shape[3];
    const index_t in_channels = in_shape[1];
    const index_t in_height = in_shape[2];
    const index_t in_width = in_shape[3];

    const index_t in_image_size = in_height * in_width;
    const index_t out_image_size = out_height * out_width;
    const index_t in_batch_size = in_channels * in_image_size;
    const index_t out_batch_size = out_channels * out_image_size;

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t c = start1; c < end1; c += step1) {
          const index_t out_base = b * out_batch_size + c * out_image_size;
          const index_t in_base = b * in_batch_size + c * in_image_size;
          const index_t in_height = in_shape[2];
          const index_t in_width = in_shape[3];
          const index_t out_height = out_shape[2];
          const index_t out_width = out_shape[3];
          for (index_t h = 0; h < out_height; ++h) {
            for (index_t w = 0; w < out_width; ++w) {
              const index_t out_offset = out_base + h * out_width + w;
              float res = 0;
              int block_size = 0;
              for (int fh = 0; fh < filter_hw[0]; ++fh) {
                for (int fw = 0; fw < filter_hw[1]; ++fw) {
                  index_t inh =
                      h * stride_hw[0] + dilation_hw[0] * fh - pad_hw[0];
                  index_t inw =
                      w * stride_hw[1] + dilation_hw[1] * fw - pad_hw[1];
                  if (inh >= 0 && inh < in_height && inw >= 0
                      && inw < in_width) {
                    index_t input_offset = in_base + inh * in_width + inw;
                    res += input[input_offset];
                    ++block_size;
                  }
                }
              }
              output[out_offset] = res / block_size;
            }
          }
        }
      }
    }, 0, batch, 1, 0, out_channels, 1);
  }
};

#ifdef MACE_ENABLE_QUANTIZE
template<>
class PoolingOp<DeviceType::CPU, uint8_t> : public PoolingOpBase {
 public:
  explicit PoolingOp(OpConstructContext *context)
      : PoolingOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input_tensor = this->Input(0);
    Tensor *output_tensor = this->Output(0);
    MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1,
               "Quantized pooling does not support dilation > 1 yet.");
    // Use the same scale and zero point with input and output.
    output_tensor->SetScale(input_tensor->scale());
    output_tensor->SetZeroPoint(input_tensor->zero_point());

    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {
        input_tensor->dim(3), kernels_[0], kernels_[1], input_tensor->dim(3)};

    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      CalcPaddingAndOutputSize(input_tensor->shape().data(),
                               DataFormat::NHWC,
                               filter_shape.data(),
                               DataFormat::OHWI,
                               dilations_.data(),
                               strides_.data(),
                               padding_type_,
                               output_shape.data(),
                               paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input_tensor->shape().data(),
                     DataFormat::NHWC,
                     filter_shape.data(),
                     DataFormat::OHWI,
                     paddings_.data(),
                     dilations_.data(),
                     strides_.data(),
                     round_type_,
                     output_shape.data());
    }
    MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

    const index_t out_channels = output_tensor->dim(3);
    const index_t in_channels = input_tensor->dim(3);
    MACE_CHECK(out_channels == in_channels);

    Tensor::MappingGuard input_guard(input_tensor);
    Tensor::MappingGuard output_guard(output_tensor);
    const uint8_t *input = input_tensor->data<uint8_t>();
    uint8_t *output = output_tensor->mutable_data<uint8_t>();
    int pad_hw[2] = {paddings[0] / 2, paddings[1] / 2};

    if (pooling_type_ == PoolingType::MAX) {
      MaxPooling(context,
                 input,
                 input_tensor->shape().data(),
                 output_shape.data(),
                 kernels_.data(),
                 strides_.data(),
                 pad_hw,
                 output);
    } else if (pooling_type_ == PoolingType::AVG) {
      AvgPooling(context,
                 input,
                 input_tensor->shape().data(),
                 output_shape.data(),
                 kernels_.data(),
                 strides_.data(),
                 pad_hw,
                 output);
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void MaxPooling(const OpContext *context,
                  const uint8_t *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *pad_hw,
                  uint8_t *output) {
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    thread_pool.Compute3D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1,
                              index_t start2, index_t end2, index_t step2) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t h = start1; h < end1; h += step1) {
          for (index_t w = start2; w < end2; w += step2) {
            const index_t out_height = out_shape[1];
            const index_t out_width = out_shape[2];
            const index_t channels = out_shape[3];
            const index_t in_height = in_shape[1];
            const index_t in_width = in_shape[2];
            const index_t in_h_base = h * stride_hw[0] - pad_hw[0];
            const index_t in_w_base = w * stride_hw[1] - pad_hw[1];
            const index_t in_h_begin = std::max<index_t>(0, in_h_base);
            const index_t in_w_begin = std::max<index_t>(0, in_w_base);
            const index_t in_h_end =
                std::min(in_height, in_h_base + filter_hw[0]);
            const index_t in_w_end =
                std::min(in_width, in_w_base + filter_hw[1]);

            uint8_t *out_ptr =
                output + ((b * out_height + h) * out_width + w) * channels;
            std::fill_n(out_ptr, channels, 0);
            for (index_t ih = in_h_begin; ih < in_h_end; ++ih) {
              for (index_t iw = in_w_begin; iw < in_w_end; ++iw) {
                const uint8_t *in_ptr = input +
                    ((b * in_height + ih) * in_width + iw) * channels;
                index_t c = 0;
#if defined(MACE_ENABLE_NEON)
                for (; c <= channels - 16; c += 16) {
                  uint8x16_t out_vec = vld1q_u8(out_ptr + c);
                  uint8x16_t in_vec = vld1q_u8(in_ptr + c);
                  out_vec = vmaxq_u8(out_vec, in_vec);
                  vst1q_u8(out_ptr + c, out_vec);
                }
                for (; c <= channels - 8; c += 8) {
                  uint8x8_t out_vec = vld1_u8(out_ptr + c);
                  uint8x8_t in_vec = vld1_u8(in_ptr + c);
                  out_vec = vmax_u8(out_vec, in_vec);
                  vst1_u8(out_ptr + c, out_vec);
                }
#endif
                for (; c < channels; ++c) {
                  out_ptr[c] = std::max(out_ptr[c], in_ptr[c]);
                }
              }
            }
          }
        }
      }
    }, 0, out_shape[0], 1, 0, out_shape[1], 1, 0, out_shape[2], 1);
  }

  void AvgPooling(const OpContext *context,
                  const uint8_t *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *pad_hw,
                  uint8_t *output) {
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    thread_pool.Compute3D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1,
                              index_t start2, index_t end2, index_t step2) {
      for (index_t b = start0; b < end0; b += step0) {
        for (index_t h = start1; h < end1; h += step1) {
          for (index_t w = start2; w < end2; w += step2) {
            const index_t out_height = out_shape[1];
            const index_t out_width = out_shape[2];
            const index_t channels = out_shape[3];
            const index_t in_height = in_shape[1];
            const index_t in_width = in_shape[2];
            const index_t in_h_base = h * stride_hw[0] - pad_hw[0];
            const index_t in_w_base = w * stride_hw[1] - pad_hw[1];
            const index_t in_h_begin = std::max<index_t>(0, in_h_base);
            const index_t in_w_begin = std::max<index_t>(0, in_w_base);
            const index_t in_h_end =
                std::min(in_height, in_h_base + filter_hw[0]);
            const index_t in_w_end =
                std::min(in_width, in_w_base + filter_hw[1]);
            const index_t block_size =
                (in_h_end - in_h_begin) * (in_w_end - in_w_begin);
            MACE_CHECK(block_size > 0);

            std::vector<uint16_t> average_buffer(channels);
            uint16_t *avg_buffer = average_buffer.data();
            std::fill_n(avg_buffer, channels, 0);
            for (index_t ih = in_h_begin; ih < in_h_end; ++ih) {
              for (index_t iw = in_w_begin; iw < in_w_end; ++iw) {
                const uint8_t *in_ptr = input +
                    ((b * in_height + ih) * in_width + iw) * channels;
                index_t c = 0;
#if defined(MACE_ENABLE_NEON)
                for (; c <= channels - 16; c += 16) {
                  uint16x8_t avg_vec[2];
                  avg_vec[0] = vld1q_u16(avg_buffer + c);
                  avg_vec[1] = vld1q_u16(avg_buffer + c + 8);
                  uint8x16_t in_vec = vld1q_u8(in_ptr + c);
                  avg_vec[0] = vaddw_u8(avg_vec[0], vget_low_u8(in_vec));
                  avg_vec[1] = vaddw_u8(avg_vec[1], vget_high_u8(in_vec));
                  vst1q_u16(avg_buffer + c, avg_vec[0]);
                  vst1q_u16(avg_buffer + c + 8, avg_vec[1]);
                }
                for (; c <= channels - 8; c += 8) {
                  uint16x8_t avg_vec = vld1q_u16(avg_buffer + c);
                  uint8x8_t in_vec = vld1_u8(in_ptr + c);
                  avg_vec = vaddw_u8(avg_vec, in_vec);
                  vst1q_u16(avg_buffer + c, avg_vec);
                }
#endif
                for (; c < channels; ++c) {
                  avg_buffer[c] += in_ptr[c];
                }
              }
            }
            uint8_t *out_ptr =
                output + ((b * out_height + h) * out_width + w) * channels;
            for (index_t c = 0; c < channels; ++c) {
              out_ptr[c] = static_cast<uint8_t>(
                  (avg_buffer[c] + block_size / 2) / block_size);
            }
          }
        }
      }
    }, 0, out_shape[0], 1, 0, out_shape[1], 1, 0, out_shape[2], 1);
  }
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class PoolingOp<DeviceType::GPU, T> : public PoolingOpBase {
 public:
  explicit PoolingOp(OpConstructContext *context)
      : PoolingOpBase(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::PoolingKernel<T>>();
    } else {
      kernel_ = make_unique<opencl::buffer::PoolingKernel<T>>();
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    return kernel_->Compute(context, input, pooling_type_, kernels_.data(),
                            strides_.data(), padding_type_, paddings_,
                            dilations_.data(), round_type_, output);
  }

 private:
  std::unique_ptr<OpenCLPoolingKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterPooling(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Pooling", PoolingOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Pooling", PoolingOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Pooling", PoolingOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Pooling", PoolingOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
