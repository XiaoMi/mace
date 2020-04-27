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

#include "mace/core/runtime/hexagon/hexagon_hta_transformer.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/quantize.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/utils/math.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_helper.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace {
template <DeviceType D>
class QuantizeTransformer;

template <>
class QuantizeTransformer<DeviceType::CPU> : public BaseTransformer {
 public:
  void Init(Device *device) override {
    device_ = device;
    quantize_util_.Init(&device_->cpu_runtime()->thread_pool());
  }
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "Quantize on CPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<float>();
    auto output_data = output->mutable_data<uint8_t>();
    quantize_util_.QuantizeWithScaleAndZeropoint(
        input_data, input->size(), input->scale(), input->zero_point(),
        output_data);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  QuantizeUtil<float, uint8_t> quantize_util_;
};

#ifdef MACE_ENABLE_OPENCL
template <>
class QuantizeTransformer<DeviceType::GPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "Quantize on GPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    const uint32_t gws = static_cast<uint32_t>(RoundUpDiv4(output->size()));
    OpenCLRuntime *runtime = device_->gpu_runtime()->opencl_runtime();
    if (kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name = MACE_OBFUSCATE_SYMBOL("buffer_quantize");
      built_options.emplace("-Dbuffer_quantize=" + kernel_name);
      built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(output->dtype()));
      MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform", kernel_name,
                                                built_options, &kernel_));
    }

    uint32_t idx = 0;
    kernel_.setArg(idx++, gws);
    kernel_.setArg(idx++, input->scale());
    kernel_.setArg(idx++, input->zero_point());
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    kernel_.setArg(idx++, *(output->opencl_buffer()));

    const uint32_t lws = static_cast<uint32_t>(
        RoundUpDiv4(runtime->GetDeviceMaxWorkGroupSize()));
    cl::Event event;
    cl_int error;
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr,
          &event);
    } else {
      uint32_t roundup_gws = RoundUp(gws, lws);
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange, cl::NDRange(roundup_gws), cl::NDRange(lws),
          nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  cl::Kernel kernel_;
};
#endif  // MACE_ENABLE_OPENCL

template <DeviceType D>
class DequantizeTransformer;

template <>
class DequantizeTransformer<DeviceType::CPU> : public BaseTransformer {
 public:
  void Init(Device *device) override {
    device_ = device;
    quantize_util_.Init(&device_->cpu_runtime()->thread_pool());
  }
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "Dequantize on CPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<uint8_t>();
    auto output_data = output->mutable_data<float>();
    quantize_util_.Dequantize(input_data, input->size(), input->scale(),
                              input->zero_point(), output_data);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  QuantizeUtil<float, uint8_t> quantize_util_;
};

#ifdef MACE_ENABLE_OPENCL
template <>
class DequantizeTransformer<DeviceType::GPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "Dequantize on GPU");
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->SetScale(input->scale());
    output->SetZeroPoint(input->zero_point());
    const uint32_t gws = static_cast<uint32_t>(RoundUpDiv4(output->size()));
    OpenCLRuntime *runtime = device_->gpu_runtime()->opencl_runtime();
    if (kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name = MACE_OBFUSCATE_SYMBOL("buffer_dequantize");
      built_options.emplace("-Dbuffer_dequantize=" + kernel_name);
      built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(output->dtype()));
      MACE_RETURN_IF_ERROR(runtime->BuildKernel("buffer_transform", kernel_name,
                                                built_options, &kernel_));
    }

    uint32_t idx = 0;
    kernel_.setArg(idx++, gws);
    kernel_.setArg(idx++, input->scale());
    kernel_.setArg(idx++, input->zero_point());
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    kernel_.setArg(idx++, *(output->opencl_buffer()));

    const uint32_t lws = static_cast<uint32_t>(
        RoundUpDiv4(runtime->GetDeviceMaxWorkGroupSize()));
    cl::Event event;
    cl_int error;
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws), nullptr,
          &event);
    } else {
      uint32_t roundup_gws = RoundUp(gws, lws);
      error = runtime->command_queue().enqueueNDRangeKernel(
          kernel_, cl::NullRange, cl::NDRange(roundup_gws), cl::NDRange(lws),
          nullptr, &event);
    }
    MACE_CL_RET_STATUS(error);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  cl::Kernel kernel_;
};
#endif  // MACE_ENABLE_OPENCL

template <DeviceType D>
class NHWCToNCHW32Transformer;
template <>
class NHWCToNCHW32Transformer<DeviceType::CPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "NHWCToNCHW32Transformer on CPU");
    int32_t padding_value = input->zero_point();
    index_t batch = input->dim(0);
    index_t height = input->dim(1);
    index_t width = input->dim(2);
    index_t channels = input->dim(3);
    index_t height_stride = width * channels;
    index_t batch_stride = height * width * channels;

    index_t output_width = RoundUp<index_t>(width, 32);
    index_t output_channel_stride = height * output_width;
    index_t output_batch_stride = channels * height * output_width;

    output->Resize({batch, channels, height, output_width});

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);
    const auto input_data = input->data<uint8_t>();
    auto output_data = output->mutable_data<uint8_t>();

    device_->cpu_runtime()->thread_pool().Compute2D(
        [=](index_t start0, index_t end0, index_t step0, index_t start1,
            index_t end1, index_t step1) {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t c = start1; c < end1; c += step1) {
              index_t input_offset = b * batch_stride + c;
              index_t output_offset =
                  b * output_batch_stride + c * output_channel_stride;
              for (index_t h = 0; h < height; ++h) {
                for (index_t w = 0; w < width; ++w) {
                  output_data[output_offset + w] =
                      input_data[input_offset + w * channels];
                }
                std::fill_n(output_data + output_offset + width,
                            output_width - width, padding_value);
                input_offset += height_stride;
                output_offset += output_width;
              }
            }
          }
        },
        0, batch, 1, 0, channels, 1);

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <>
class NHWCToNCHW32Transformer<DeviceType::GPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "NHWCToNCHW32Transformer on GPU");
    const index_t batch = input->dim(0);
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t channels = input->dim(3);
    const index_t output_width = RoundUp<index_t>(width, 32);
    std::vector<index_t> transformed_shape = {batch, channels, height,
                                              output_width};
    uint32_t gws[3];
    gws[0] = static_cast<uint32_t>(RoundUpDiv4<index_t>(output_width));
    gws[1] = static_cast<uint32_t>(height);
    gws[2] = static_cast<uint32_t>(batch * channels);
    MACE_RETURN_IF_ERROR(output->Resize(transformed_shape));

    if (kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name =
          MACE_OBFUSCATE_SYMBOL("transform_nhwc_to_nchw32");
      built_options.emplace("-Dtransform_nhwc_to_nchw32=" + kernel_name);
      std::string data_dt = DtToCLDt(input->dtype());
      built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
      built_options.emplace("-DDATA_TYPE=" + data_dt);
      MACE_RETURN_IF_ERROR(
          device_->gpu_runtime()->opencl_runtime()->BuildKernel(
              "buffer_transform", kernel_name, built_options, &kernel_));
    }
    uint32_t idx = 0;
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    kernel_.setArg(idx++, input->zero_point());
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    kernel_.setArg(idx++, static_cast<int32_t>(batch));
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));

    std::string tuning_key = Concat("transform_nhwc_to_nchw32",
                                    transformed_shape[0], transformed_shape[1],
                                    transformed_shape[2], transformed_shape[3]);
    std::vector<uint32_t> lws = {4, 4, 4, 0};
    MACE_RETURN_IF_ERROR(
        TuningOrRun3DKernel(device_->gpu_runtime()->opencl_runtime(), kernel_,
                            tuning_key, gws, lws, nullptr));

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  cl::Kernel kernel_;
};
#endif  // MACE_ENABLE_OPENCL

template <DeviceType D>
class NHWCToD32Transformer;

template <>
class NHWCToD32Transformer<DeviceType::CPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "NHWCToD32Transformer on CPU");
    index_t batch = input->dim(0);
    index_t height = input->dim(1);
    index_t width = input->dim(2);
    index_t channels = input->dim(3);
    index_t height_stride = width * channels;
    index_t batch_stride = height * width * channels;

    index_t channel_slices = RoundUpDiv(channels, static_cast<index_t>(32));
    index_t output_channel_slices_stride = width * 32;
    index_t output_height_stride = channel_slices * width * 32;
    index_t output_batch_stride = height * channel_slices * width * 32;

    std::vector<index_t> output_shape{batch, height, channel_slices, width, 32};
    output->Resize(output_shape);

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<uint8_t>();
    auto output_data = output->mutable_data<uint8_t>();
    std::fill_n(output_data, output->size(), input->zero_point());

    device_->cpu_runtime()->thread_pool().Compute2D(
        [=](index_t start0, index_t end0, index_t step0, index_t start1,
            index_t end1, index_t step1) {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t h = start1; h < end1; h += step1) {
              index_t input_offset = b * batch_stride + h * height_stride;
              index_t output_offset =
                  b * output_batch_stride + h * output_height_stride;
              for (index_t w = 0; w < width; ++w) {
                for (index_t c = 0; c < channels; ++c) {
                  output_data[output_offset +
                              c / 32 * output_channel_slices_stride + c % 32] =
                      input_data[input_offset + c];
                }
                input_offset += channels;
                output_offset += 32;
              }
            }
          }
        },
        0, batch, 1, 0, height, 1);
    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <>
class NHWCToD32Transformer<DeviceType::GPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "D32ToNHWCTransformer on GPU");
    const index_t batch = input->dim(0);
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t channels = input->dim(3);
    const index_t channel_slices = RoundUpDiv<index_t>(channels, 32);
    std::vector<index_t> output_shape{batch, height, channel_slices, width, 32};
    output->Resize(output_shape);

    uint32_t gws[3];
    gws[0] = static_cast<uint32_t>(RoundUpDiv4<index_t>(width * 32));
    gws[1] = static_cast<uint32_t>(channel_slices);
    gws[2] = static_cast<uint32_t>(batch * height);

    if (kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_nhwc_to_d32");
      built_options.emplace("-Dtransform_nhwc_to_d32=" + kernel_name);
      std::string data_dt = DtToCLDt(input->dtype());
      built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
      built_options.emplace("-DDATA_TYPE=" + data_dt);
      MACE_RETURN_IF_ERROR(
          device_->gpu_runtime()->opencl_runtime()->BuildKernel(
              "buffer_transform", kernel_name, built_options, &kernel_));
    }

    uint32_t idx = 0;
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    kernel_.setArg(idx++, input->zero_point());
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    kernel_.setArg(idx++, static_cast<int32_t>(batch));
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_slices));


    std::string tuning_key =
        Concat("transform_nhwc_to_d32", batch, height, width, channels);
    std::vector<uint32_t> lws = {4, 4, 4, 0};
    MACE_RETURN_IF_ERROR(
        TuningOrRun3DKernel(device_->gpu_runtime()->opencl_runtime(), kernel_,
                            tuning_key, gws, lws, nullptr));
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  cl::Kernel kernel_;
};
#endif  // MACE_ENABLE_OPENCL

template <DeviceType D>
class D32ToNHWCTransformer;

template <>
class D32ToNHWCTransformer<DeviceType::CPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "D32ToNHWCTransformer on CPU");
    index_t batch = output->dim(0);
    index_t height = output->dim(1);
    index_t width = output->dim(2);
    index_t channel = output->dim(3);
    index_t height_stride = width * channel;
    index_t batch_stride = height * width * channel;

    index_t channel_slices = RoundUpDiv(channel, static_cast<index_t>(32));
    index_t input_channel_slices_stride = width * 32;
    index_t input_height_stride = channel_slices * width * 32;
    index_t input_batch_stride = height * channel_slices * width * 32;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<uint8_t>();
    auto output_data = output->mutable_data<uint8_t>();

    device_->cpu_runtime()->thread_pool().Compute2D(
        [=](index_t start0, index_t end0, index_t step0, index_t start1,
            index_t end1, index_t step1) {
          for (index_t b = start0; b < end0; b += step0) {
            for (index_t h = start1; h < end1; h += step1) {
              index_t input_offset =
                  b * input_batch_stride + h * input_height_stride;
              index_t output_offset = b * batch_stride + h * height_stride;
              for (index_t w = 0; w < width; ++w) {
                for (index_t c = 0; c < channel; ++c) {
                  output_data[output_offset + c] =
                      input_data[input_offset +
                                 c / 32 * input_channel_slices_stride + c % 32];
                }
                input_offset += 32;
                output_offset += channel;
              }
            }
          }
        },
        0, batch, 1, 0, height, 1);
    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <>
class D32ToNHWCTransformer<DeviceType::GPU> : public BaseTransformer {
 public:
  MaceStatus Compute(const Tensor *input, Tensor *output) override {
    MACE_LATENCY_LOGGER(1, "D32ToNHWCTransformer on GPU");
    const index_t batch = output->dim(0);
    const index_t height = output->dim(1);
    const index_t width = output->dim(2);
    const index_t channels = output->dim(3);
    const index_t channel_slices = RoundUpDiv<index_t>(channels, 32);

    uint32_t gws[3];
    gws[0] = static_cast<uint32_t>(RoundUpDiv4<index_t>(channels));
    gws[1] = static_cast<uint32_t>(width);
    gws[2] = static_cast<uint32_t>(batch * height);

    if (kernel_.get() == nullptr) {
      std::set<std::string> built_options;
      std::string kernel_name = MACE_OBFUSCATE_SYMBOL("transform_d32_to_nhwc");
      built_options.emplace("-Dtransform_d32_to_nhwc=" + kernel_name);
      std::string data_dt = DtToCLDt(input->dtype());
      built_options.emplace("-DIN_DATA_TYPE=" + data_dt);
      built_options.emplace("-DDATA_TYPE=" + data_dt);
      MACE_RETURN_IF_ERROR(
          device_->gpu_runtime()->opencl_runtime()->BuildKernel(
              "buffer_transform", kernel_name, built_options, &kernel_));
    }

    uint32_t idx = 0;
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    MACE_CHECK(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    kernel_.setArg(idx++, *(output->opencl_buffer()));
    kernel_.setArg(idx++, static_cast<int32_t>(batch));
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_slices));


    std::string tuning_key =
        Concat("transform_d32_to_nhwc", batch, height, width, channels);
    std::vector<uint32_t> lws = {4, 4, 4, 0};
    MACE_RETURN_IF_ERROR(
        TuningOrRun3DKernel(device_->gpu_runtime()->opencl_runtime(), kernel_,
                            tuning_key, gws, lws, nullptr));
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  cl::Kernel kernel_;
};
#endif  // MACE_ENABLE_OPENCL
}  // namespace

template <DeviceType D>
void HexagonHTATranformer<D>::Init(Device *device) {
  device_ = device;
  quantizer_ = make_unique<QuantizeTransformer<D>>();
  quantizer_->Init(device);
  dequantizer_ = make_unique<DequantizeTransformer<D>>();
  dequantizer_->Init(device);
}

template <DeviceType D>
MaceStatus HexagonHTATranformer<D>::SetInputTransformer(
    hexagon_hta_hw_layout format) {
  switch (format) {
    case HEXAGON_HTA_HW_FORMAT_D32:
      input_transformers_.push_back(make_unique<NHWCToD32Transformer<D>>());
      break;
    case HEXAGON_HTA_HW_FORMAT_PLANAR:
      input_transformers_.push_back(make_unique<NHWCToNCHW32Transformer<D>>());
      break;
    case HEXAGON_HTA_HW_FORMAT_DEPTH_FIRST:
    default:
      MACE_NOT_IMPLEMENTED;
      break;
  }
  input_transformers_.back()->Init(device_);
  return MaceStatus::MACE_SUCCESS;
}

template <DeviceType D>
MaceStatus HexagonHTATranformer<D>::SetOutputTransformer(
    hexagon_hta_hw_layout format) {
  switch (format) {
    case HEXAGON_HTA_HW_FORMAT_D32:
      output_transformers_.push_back(make_unique<D32ToNHWCTransformer<D>>());
      break;
    case HEXAGON_HTA_HW_FORMAT_PLANAR:
    case HEXAGON_HTA_HW_FORMAT_DEPTH_FIRST:
    default:
      MACE_NOT_IMPLEMENTED;
      break;
  }
  output_transformers_.back()->Init(device_);
  return MaceStatus::MACE_SUCCESS;
}

template <DeviceType D>
MaceStatus HexagonHTATranformer<D>::TransformInput(const Tensor *input,
                                                   Tensor *output,
                                                   int index) {
  return input_transformers_[index]->Compute(input, output);
}

template <DeviceType D>
MaceStatus HexagonHTATranformer<D>::TransformOutput(const Tensor *input,
                                                    Tensor *output,
                                                    int index) {
  return output_transformers_[index]->Compute(input, output);
}

template <DeviceType D>
MaceStatus HexagonHTATranformer<D>::Quantize(const Tensor *input,
                                             Tensor *output) {
  return quantizer_->Compute(input, output);
}

template <DeviceType D>
MaceStatus HexagonHTATranformer<D>::Dequantize(const Tensor *input,
                                               Tensor *output) {
  return dequantizer_->Compute(input, output);
}

template void HexagonHTATranformer<CPU>::Init(Device *device);
template MaceStatus HexagonHTATranformer<CPU>::Quantize(const Tensor *input,
                                                        Tensor *output);
template MaceStatus HexagonHTATranformer<CPU>::Dequantize(const Tensor *input,
                                                          Tensor *output);
template MaceStatus HexagonHTATranformer<CPU>::SetInputTransformer(
    hexagon_hta_hw_layout format);
template MaceStatus HexagonHTATranformer<CPU>::SetOutputTransformer(
    hexagon_hta_hw_layout format);
template MaceStatus HexagonHTATranformer<CPU>::TransformInput(
    const Tensor *input, Tensor *output, int index);
template MaceStatus HexagonHTATranformer<CPU>::TransformOutput(
    const Tensor *input, Tensor *output, int index);

#ifdef MACE_ENABLE_OPENCL
template void HexagonHTATranformer<GPU>::Init(Device *device);
template MaceStatus HexagonHTATranformer<GPU>::Quantize(const Tensor *input,
                                                        Tensor *output);
template MaceStatus HexagonHTATranformer<GPU>::Dequantize(const Tensor *input,
                                                          Tensor *output);
template MaceStatus HexagonHTATranformer<GPU>::SetInputTransformer(
    hexagon_hta_hw_layout format);
template MaceStatus HexagonHTATranformer<GPU>::SetOutputTransformer(
    hexagon_hta_hw_layout format);
template MaceStatus HexagonHTATranformer<GPU>::TransformInput(
    const Tensor *input, Tensor *output, int index);
template MaceStatus HexagonHTATranformer<GPU>::TransformOutput(
    const Tensor *input, Tensor *output, int index);
#endif  // MACE_ENABLE_OPENCL
}  // namespace mace
