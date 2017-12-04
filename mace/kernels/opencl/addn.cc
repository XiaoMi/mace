//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/addn.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
static void AddN(const std::vector<const Tensor *> &input_tensors,
                 Tensor *output) {
  if (input_tensors.size() > 4) {
    MACE_NOT_IMPLEMENTED;
  }
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t width_pixels = channel_blocks * width;
  const index_t batch_height_pixels = batch * height;

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  built_options.emplace("-DINPUT_NUM=" + ToString(input_tensors.size()));
  auto addn_kernel = runtime->BuildKernel("addn", "addn", built_options);

  const uint32_t lws = runtime->GetKernelMaxWorkGroupSize(addn_kernel);

  uint32_t idx = 0;
  for (auto input : input_tensors) {
  addn_kernel.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(input->buffer())));
  }
  addn_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));

  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      addn_kernel, cl::NullRange,
      cl::NDRange(width_pixels, batch_height_pixels),
      cl::NDRange(64, 16),  // TODO fix this
      nullptr, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS) << "error code: " << error;
}

template <typename T>
void AddNFunctor<DeviceType::OPENCL, T>::operator()(
    const std::vector<const Tensor *> &input_tensors, Tensor *output_tensor) {
  size_t size = input_tensors.size();
  MACE_CHECK(size >= 2 && input_tensors[0] != nullptr);

  const index_t batch = input_tensors[0]->dim(0);
  const index_t height = input_tensors[0]->dim(1);
  const index_t width = input_tensors[0]->dim(2);
  const index_t channels = input_tensors[0]->dim(3);

  for (int i = 1; i < size; ++i) {
    MACE_CHECK_NOTNULL(input_tensors[i]);
    MACE_CHECK(batch == input_tensors[i]->dim(0));
    MACE_CHECK(height == input_tensors[i]->dim(1));
    MACE_CHECK(width == input_tensors[i]->dim(2));
    MACE_CHECK(channels == input_tensors[i]->dim(3));
  }

  std::vector<index_t> output_shape = input_tensors[0]->shape();
  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
  output_tensor->ResizeImage(output_shape, output_image_shape);

  AddN<T>(input_tensors, output_tensor);
};

template
struct AddNFunctor<DeviceType::OPENCL, float>;

template
struct AddNFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  //  namespace mace
