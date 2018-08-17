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

#include "mace/kernels/deconv_2d.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

namespace {

MaceStatus Deconv2dOpencl(cl::Kernel *kernel,
                          const Tensor *input,
                          const Tensor *filter,
                          const Tensor *bias,
                          const int stride,
                          const int *paddings,
                          const ActivationType activation,
                          const float relux_max_limit,
                          const DataType dt,
                          std::vector<index_t> *prev_input_shape,
                          Tensor *output,
                          StatsFuture *future,
                          uint32_t *kwg_size,
                          std::unique_ptr<BufferBase> *kernel_error) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  MACE_CHECK(stride > 0, "stride should > 0.");
#define MACE_WIDTH_BLK 5
  const index_t n_strides = (width + stride - 1) / stride;
  const index_t width_blocks =
      ((n_strides + MACE_WIDTH_BLK - 1) / MACE_WIDTH_BLK) * stride;
  const float stride_r = 1.f / static_cast<float>(stride);
  const int padding_h = (paddings[0] + 1) >> 1;
  const int padding_w = (paddings[0] + 1) >> 1;

  const int align_h = stride - 1 - padding_h;
  const int align_w = stride - 1 - padding_w;
  const int kernel_size = filter->dim(2) * filter->dim(3);

  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(*kernel_error);
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("deconv_2d");
    built_options.emplace("-Ddeconv_2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("deconv_2d", kernel_name,
                                              built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG_PTR;
    SET_3D_GWS_ARGS_PTR(kernel, gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel->setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel->setArg(idx++, static_cast<int32_t>(input->dim(3)));
    kernel->setArg(idx++, static_cast<int32_t>(height));
    kernel->setArg(idx++, static_cast<int32_t>(width));
    kernel->setArg(idx++, static_cast<int32_t>(channels));
    kernel->setArg(idx++, static_cast<int32_t>(stride));
    kernel->setArg(idx++, stride_r);
    kernel->setArg(idx++, static_cast<int32_t>(align_h));
    kernel->setArg(idx++, static_cast<int32_t>(align_w));
    kernel->setArg(idx++, static_cast<int32_t>(padding_h));
    kernel->setArg(idx++, static_cast<int32_t>(padding_w));
    kernel->setArg(idx++, static_cast<int32_t>(filter->dim(2)));
    kernel->setArg(idx++, static_cast<int32_t>(filter->dim(3)));
    kernel->setArg(idx++, static_cast<int32_t>(kernel_size));
    kernel->setArg(idx++, static_cast<int32_t>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<int32_t>(channel_blocks));

    *prev_input_shape = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(gws, *kwg_size);
  std::string tuning_key =
      Concat("deconv2d_opencl_kernel_", activation, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(*kernel, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(*kernel_error);
  return MACE_SUCCESS;
}

}  // namespace

template <typename T>
MaceStatus Deconv2dFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    const Tensor *output_shape_tensor,
    Tensor *output,
    StatsFuture *future) {
  MACE_CHECK_NOTNULL(input);
  MACE_CHECK_NOTNULL(filter);
  MACE_CHECK_NOTNULL(output);
  if (!from_caffe_) {
    if (output_shape_.size() != 4) {
      MACE_CHECK_NOTNULL(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data =
          output_shape_tensor->data<int32_t>();
      output_shape_ =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);
    }
    paddings_.clear();
    paddings_ = std::vector<int>(2, 0);
    CalcDeconvPaddingAndInputSize(input->shape().data(), filter->shape().data(),
                                  strides_, padding_type_, output_shape_.data(),
                                  paddings_.data());
  } else {
    output_shape_.clear();
    output_shape_ = std::vector<index_t>(4, 0);
    CalcDeconvOutputSize(input->shape().data(), filter->shape().data(),
                         strides_, output_shape_.data(), paddings_.data());
  }
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape_, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape_, output_image_shape));

  return Deconv2dOpencl(&kernel_, input, filter, bias, strides_[0],
                        paddings_.data(), activation_, relux_max_limit_,
                        DataTypeToEnum<T>::value, &input_shape_, output, future,
                        &kwg_size_, &kernel_error_);
}

template struct Deconv2dFunctor<DeviceType::GPU, float>;
template struct Deconv2dFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
