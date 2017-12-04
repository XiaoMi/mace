//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/concat.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

static void Concat2(const Tensor *input0,
                    const Tensor *input1,
                    const DataType dt,
                    Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  const int channel_blk = RoundUpDiv4(channel);

  auto runtime = OpenCLRuntime::Get();
  std::set<std::string> built_options;
  if (input0->dtype() == output->dtype()) {
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
  } else {
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  }
  if (input0->dim(3) % 4 == 0) {
    built_options.emplace("-DDIVISIBLE_FOUR");
  }
  auto concat_kernel = runtime->BuildKernel("concat", "concat_channel", built_options);

  uint32_t idx = 0;
  concat_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input0->buffer())));
  concat_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input1->buffer())));
  concat_kernel.setArg(idx++, static_cast<int32_t>(input0->dim(3)));
  concat_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(concat_kernel);

  uint32_t lws[3];
  lws[0] = std::min<uint32_t>(channel_blk, kwg_size);
  lws[1] = std::min<uint32_t>(width, kwg_size / lws[0]);
  lws[2] = std::min<uint32_t>(height * batch, kwg_size / (lws[0] * lws[1]));

  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      concat_kernel, cl::NullRange,
      cl::NDRange(static_cast<uint32_t>(channel_blk),
                  static_cast<uint32_t>(width),
                  static_cast<uint32_t>(height * batch)),
      cl::NDRange(lws[0], lws[1], lws[2]),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS);
}

template<typename T>
void ConcatFunctor<DeviceType::OPENCL, T>::operator()(const std::vector<const Tensor *> &input_list,
                                                      Tensor *output) {
  const int inputs_count = input_list.size();
  MACE_CHECK(inputs_count == 2 && axis_ == 3)
    << "Concat opencl kernel only support two elements with axis == 3";

  const Tensor *input0 = input_list[0];

  std::vector<index_t> output_shape(input0->shape());
  for (int i = 1; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    MACE_CHECK(input->dim_size() == input0->dim_size(),
               "Ranks of all input tensors must be same.");
    for (int j = 0; j < input->dim_size(); ++j) {
      if (j == axis_) {
        continue;
      }
      MACE_CHECK(input->dim(j) == input0->dim(j),
                 "Dimensions of inputs should equal except axis.");
    }
    output_shape[axis_] += input->dim(axis_);
  }
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, image_shape);
  output->ResizeImage(output_shape, image_shape);

  switch (inputs_count) {
    case 2:
      Concat2(input_list[0], input_list[1], DataTypeToEnum<T>::value, output);
      break;
    default:MACE_NOT_IMPLEMENTED;
  }
};

template
struct ConcatFunctor<DeviceType::OPENCL, float>;
template
struct ConcatFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
} //  namespace mace
