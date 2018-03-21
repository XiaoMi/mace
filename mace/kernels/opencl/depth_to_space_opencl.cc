//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/depth_to_space.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void DepthToSpaceOpFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    Tensor *output,
    StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t input_h = input->dim(1);
  const index_t input_w = input->dim(2);
  const index_t input_d = input->dim(3);
    
  const index_t output_h = input_h * block_size_;
  const index_t output_w = input_w * block_size_;
  const index_t output_d = input_d / (block_size_ * block_size_);
    
  std::vector<index_t> output_shape = {batch, output_h, output_w, output_d};  
  
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);  
  output->ResizeImage(output_shape, image_shape);
  
  const int output_depth_blocks = RoundUpDiv4(output_d); 

  if (kernel_.get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("depth_to_space");
    built_options.emplace("-Ddepth_to_space=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    kernel_ = runtime->BuildKernel("depth_to_space", kernel_name,
                                   built_options);
  }
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, block_size_);
    kernel_.setArg(idx++, output_depth_blocks);
    kernel_.setArg(idx++, *(output->opencl_image()));
    
    input_shape_ = input->shape();
  }
  
  const uint32_t gws[3] = {static_cast<uint32_t>(output_depth_blocks),
                           static_cast<uint32_t>(output_w),
                           static_cast<uint32_t>(output_h * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::stringstream ss;
  ss << "depth_to_space_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
}

template
struct DepthToSpaceOpFunctor<DeviceType::OPENCL, float>;
template
struct DepthToSpaceOpFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
