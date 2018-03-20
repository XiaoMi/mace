//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/pooling.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void PoolingFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                       Tensor *output,
                                                       StatsFuture *future) {
  MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1)
      << "Pooling opencl kernel not support dilation yet";

  if (kernel_.get() == nullptr) {
    const DataType dt = DataTypeToEnum<T>::value;
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pooling");
    built_options.emplace("-Dpooling=" + kernel_name);
    if (pooling_type_ == MAX && input->dtype() == output->dtype()) {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
      built_options.emplace(dt == DT_HALF ? "-DFP16" : "");
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
      built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    }
    if (pooling_type_ == AVG) {
      built_options.emplace("-DPOOL_AVG");
    }
    kernel_ = runtime->BuildKernel("pooling", kernel_name, built_options);
  }
  if (!IsVecEqual(input_shape_, input->shape())) {
    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {kernels_[0], kernels_[1],
                                         input->dim(3), input->dim(3)};

    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      kernels::CalcNHWCPaddingAndOutputSize(
          input->shape().data(), filter_shape.data(), dilations_, strides_,
          padding_type_, output_shape.data(), paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input->shape().data(), filter_shape.data(),
                     paddings_.data(), dilations_, strides_, RoundType::CEIL,
                     output_shape.data());
    }

    std::vector<size_t> output_image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                    &output_image_shape);
    output->ResizeImage(output_shape, output_image_shape);

    uint32_t idx = 0;
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(output->dim(1)));
    kernel_.setArg(idx++, paddings[0] / 2);
    kernel_.setArg(idx++, paddings[1] / 2);
    kernel_.setArg(idx++, strides_[0]);
    kernel_.setArg(idx++, kernels_[0]);
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  index_t batch = output->dim(0);
  index_t out_height = output->dim(1);
  index_t out_width = output->dim(2);
  index_t channels = output->dim(3);

  index_t channel_blocks = (channels + 3) / 4;


  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blocks), static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(batch * out_height),
  };
  std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::stringstream ss;
  ss << "pooling_opencl_kernel_" << output->dim(0) << "_" << output->dim(1)
     << "_" << output->dim(2) << "_" << output->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct PoolingFunctor<DeviceType::OPENCL, float>;
template struct PoolingFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace
