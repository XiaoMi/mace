#include "mace/ops/opencl/image/argmax.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ArgMaxKernel::Compute(OpContext *context, const Tensor *input, Tensor *output) {

  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  std::vector<index_t> output_shape = input->shape();
  output_shape[3] = 1;

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("argmax");
    built_options.emplace("-Dargmax=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("argmax", kernel_name, built_options, &kernel_));

    kwg_size_ = static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {1, static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  MACE_OUT_OF_RANGE_INIT(kernel_);       
  if (!IsVecEqual(input_shape_, input->shape())) {

    std::vector<size_t> output_image_shape;
    OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL, &output_image_shape);
    MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

    int idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, static_cast<uint32_t>(channel_blocks));
    kernel_.setArg(idx++, *(output->opencl_image()));
    input_shape_ = input->shape();
  }             

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key = Concat("argmax_opencl_kernel", output->dim(0), output->dim(1), output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key, gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace