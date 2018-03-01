//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/concat.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

static void Concat2(cl::Kernel *kernel,
                    const Tensor *input0,
                    const Tensor *input1,
                    const DataType dt,
                    Tensor *output,
                    StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  const int channel_blk = RoundUpDiv4(channel);

  if (kernel->get() == nullptr) {
    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("concat_channel");
    built_options.emplace("-Dconcat_channel=" + kernel_name);
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
    *kernel = runtime->BuildKernel("concat", kernel_name, built_options);

    uint32_t idx = 0;
    kernel->setArg(idx++, *(static_cast<const cl::Image2D *>(input0->buffer())));
    kernel->setArg(idx++, *(static_cast<const cl::Image2D *>(input1->buffer())));
    kernel->setArg(idx++, static_cast<int32_t>(input0->dim(3)));
    kernel->setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  }

  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk),
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(batch * height),
  };
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::stringstream ss;
  ss << "concat_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  TuningOrRun3DKernel(*kernel, ss.str(), gws, lws, future);
}

template<typename T>
void ConcatFunctor<DeviceType::OPENCL, T>::operator()(const std::vector<const Tensor *> &input_list,
                                                      Tensor *output,
                                                      StatsFuture *future) {
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
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, image_shape);
  output->ResizeImage(output_shape, image_shape);

  switch (inputs_count) {
    case 2:
      Concat2(&kernel_, input_list[0], input_list[1], DataTypeToEnum<T>::value,
              output, future);
      break;
    default:MACE_NOT_IMPLEMENTED;
  }
};

template
struct ConcatFunctor<DeviceType::OPENCL, float>;
template
struct ConcatFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
