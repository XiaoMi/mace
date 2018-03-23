//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/space_to_depth.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void SpaceToDepthOpFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    Tensor *output,
    StatsFuture *future) {
  const index_t batch_size = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_depth = input->dim(3);
  
  const index_t output_height = input_height / block_size_;
  const index_t output_width = input_width / block_size_;
  const index_t output_depth = input_depth * block_size_ * block_size_;
  
  std::vector<index_t> output_shape = {batch_size, output_height, output_width, output_depth};  
  
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);  
  output->ResizeImage(output_shape, image_shape);
  
  const int input_depth_blocks = RoundUpDiv4(input_depth); 

  if (kernel_.get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("space_to_depth");
    built_options.emplace("-Dspace_to_depth=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    kernel_ = runtime->BuildKernel("space_to_depth", kernel_name,
                                   built_options);
  }
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, block_size_);
    kernel_.setArg(idx++, input_depth_blocks);
    kernel_.setArg(idx++, *(output->opencl_image()));
    
    input_shape_ = input->shape();
  }
  
  const uint32_t gws[3] = {static_cast<uint32_t>(input_depth_blocks),
                           static_cast<uint32_t>(input_width),
                           static_cast<uint32_t>(input_height * batch_size)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::stringstream ss;
  ss << "space_to_depth_opencl_kernel_"
     << input->dim(0) << "_"
     << input->dim(1) << "_"
     << input->dim(2) << "_"
     << input->dim(3);
  TuningOrRun3DKernel(kernel_, ss.str(), gws, lws, future);
}

template
struct SpaceToDepthOpFunctor<DeviceType::OPENCL, float>;
template
struct SpaceToDepthOpFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
