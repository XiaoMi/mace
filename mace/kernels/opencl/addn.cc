//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/addn.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

static void Add2(const Tensor *input0, const Tensor *input1, Tensor *output) {
  index_t element_size = input0->NumElements();
  index_t blocks = (element_size + 3) / 4;

  const uint32_t gws = blocks;

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(output->dtype()));
  auto addn_kernel = runtime->BuildKernel("addn", "add2", built_options);

  const uint32_t lws = runtime->GetKernelMaxWorkGroupSize(addn_kernel);

  uint32_t idx = 0;
  addn_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input0->buffer())));
  addn_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input1->buffer())));
  addn_kernel.setArg(idx++, static_cast<int32_t>(element_size));
  addn_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      addn_kernel, cl::NullRange,
      cl::NDRange(gws),
      cl::NDRange(lws),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS);
}

template<>
void AddNFunctor<DeviceType::OPENCL, float>::operator()(std::vector<const Tensor *> &input_tensors,
                                                        Tensor *output_tensor) {

  if (input_tensors.empty() || input_tensors.front() == nullptr) {
    return;
  }
  size_t size = input_tensors.size();

  switch (size) {
    case 2:Add2(input_tensors[0], input_tensors[1], output_tensor);
      break;
    default:MACE_NOT_IMPLEMENTED;
  }
};

}  // namespace kernels
} //  namespace mace
