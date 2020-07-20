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

// This Op is for fused StatisticsExtraction and StatisticsPooling
// Components in Kaldi.
// This op is used to extract moving-average mean and standard-deviation
// statistics of input data.
// 'forward_indexes' indicates which frames of input will be used for
// extraction.
// save statistics results.
// 'forward_indexes' and 'count' were from precomputed index in kaldi.
// Reference to tools/extract_pooling.py and
// http://kaldi-asr.org/doc/nnet-general-component_8h_source.html#l00158

#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/tensor.h"
#include "mace/ops/conv_pool_2d_base.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/extract_image_patches.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class ExtractImagePatchesOp;

template<class T>
class ExtractImagePatchesOp<DeviceType::CPU, T> : public ConvPool2dOpBase {
 public:
  explicit ExtractImagePatchesOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        kernels_(Operation::GetRepeatedArgs<int>("kernels")) {}

  MaceStatus Run(OpContext *context) override {
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
      CalcNCHWOutputSize(input_tensor->shape().data(), filter_shape.data(),
                         paddings_.data(), dilations_.data(), strides_.data(),
                         RoundType::FLOOR, output_shape.data());
    }
    output_shape[1] *= kernels_[0] * kernels_[1];

    MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

    Tensor::MappingGuard input_guard(input_tensor);
    Tensor::MappingGuard output_guard(output_tensor);
    const T *input = input_tensor->data<T>();
    MACE_CHECK(output_tensor->dtype() == DataTypeToEnum<T>::value);
    T *output = output_tensor->mutable_data<T>();
    const index_t *input_shape = input_tensor->shape().data();
    int pad_hw[2] = {paddings[0] / 2, paddings[1] / 2};

    return ExtractImagePatches(context, input, input_shape, output_shape.data(),
                               kernels_.data(), strides_.data(),
                               dilations_.data(), pad_hw, output);
  }

 private:
  MaceStatus ExtractImagePatches(const OpContext *context,
                                 const T *input,
                                 const index_t *in_shape,
                                 const index_t *out_shape,
                                 const int *filter_hw,
                                 const int *stride_hw,
                                 const int *dilation_hw,
                                 const int *pad_hw,
                                 T *output) {
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
          const index_t in_c = c % in_channels;
          const index_t filter_idx = c / in_channels;
          const index_t out_base = b * out_batch_size + c * out_image_size;
          const index_t in_base = b * in_batch_size + in_c * in_image_size;

          for (index_t h = 0; h < out_height; ++h) {
            for (index_t w = 0; w < out_width; ++w) {
              index_t out_offset = out_base + h * out_width + w;
              index_t fh = filter_idx / filter_hw[1];
              index_t fw = filter_idx % filter_hw[1];
              index_t inh = h * stride_hw[0] + dilation_hw[0] * fh - pad_hw[0];
              index_t inw = w * stride_hw[1] + dilation_hw[1] * fw - pad_hw[1];
              if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                index_t input_offset = in_base + inh * in_width + inw;
                output[out_offset] = input[input_offset];
              } else {
                output[out_offset] = 0;
              }
            }
          }
        }
      }
    }, 0, batch, 1, 0, out_channels, 1);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::vector<int> kernels_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template<>
class ExtractImagePatchesOp<DeviceType::GPU, float> : public ConvPool2dOpBase {
 public:
  explicit ExtractImagePatchesOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        kernels_(Operation::GetRepeatedArgs<int>("kernels")) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ExtractImagePatchesKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    return kernel_->Compute(context, input, kernels_.data(), strides_.data(),
                            padding_type_, paddings_, dilations_.data(),
                            output);
  }

 private:
  std::vector<int> kernels_;
  std::unique_ptr<OpenCLExtractImagePatchesKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterExtractImagePatches(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "ExtractImagePatches", ExtractImagePatchesOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "ExtractImagePatches",
                        ExtractImagePatchesOp, DeviceType::CPU);
  MACE_REGISTER_GPU_OP(op_registry, "ExtractImagePatches",
                       ExtractImagePatchesOp);

  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("ExtractImagePatches").SetDevicePlacerFunc(
          [](OpConditionContext *context) -> std::set<DeviceType> {
            auto op = context->operator_def();
            if (op->output_shape_size() != op->output_size()) {
              return {DeviceType::CPU, DeviceType::GPU};
            }
            auto kernels = ProtoArgHelper::GetRepeatedArgs<OperatorDef, int>(
                *op, "kernels");
            auto &output_shape = op->output_shape(0);
            auto &output_dims = output_shape.dims();
            auto in_channel = output_dims[3] / kernels[0] / kernels[1];
            if (output_shape.dims_size() != 4 || in_channel % 4 != 0) {
              return {DeviceType::CPU};
            }
#ifdef MACE_ENABLE_OPENCL
            if (context->device()->device_type() == DeviceType::GPU) {
              auto opencl_runtime =
                  context->device()->gpu_runtime()->opencl_runtime();
              auto max_2d_size = opencl_runtime->GetMaxImage2DSize();
              auto image_width = output_dims[2] * output_dims[3] / 4;
              if (image_width > static_cast<index_t>(max_2d_size[0])) {
                return {DeviceType::CPU};
              }
            }
#endif  // MACE_ENABLE_OPENCL

            return {DeviceType::CPU, DeviceType::GPU};
          }));
}

}  // namespace ops
}  // namespace mace
