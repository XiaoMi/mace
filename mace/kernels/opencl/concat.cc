//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/concat.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

static void Concat2(cl::Kernel *kernel,
                    const Tensor *input0,
                    const Tensor *input1,
                    const DataType dt,
                    std::vector<index_t> *prev_input_shape,
                    Tensor *output,
                    StatsFuture *future,
                    uint32_t *kwg_size,
                    std::unique_ptr<BufferBase> *kernel_error) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  const int channel_blk = RoundUpDiv4(channel);
  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk), static_cast<uint32_t>(width),
      static_cast<uint32_t>(batch * height),
  };

  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("concat_channel");
    built_options.emplace("-Dconcat_channel=" + kernel_name);
    if (runtime->IsOutOfRangeCheckEnabled()) {
      built_options.emplace("-DOUT_OF_RANGE_CHECK");
      *kernel_error = std::move(std::unique_ptr<Buffer>(
            new Buffer(GetDeviceAllocator(DeviceType::OPENCL), 1)));
      (*kernel_error)->Map(nullptr);
      *((*kernel_error)->mutable_data<char>()) = '0';
      (*kernel_error)->UnMap();
    }
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
    }
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

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }
  if (!IsVecEqual(*prev_input_shape, input0->shape())) {
    uint32_t idx = 0;
    if (runtime->IsOutOfRangeCheckEnabled()) {
      kernel->setArg(idx++,
          *(static_cast<cl::Buffer *>((*kernel_error)->buffer())));
    }
    if (!runtime->IsNonUniformWorkgroupsSupported()) {
      kernel->setArg(idx++, gws[0]);
      kernel->setArg(idx++, gws[1]);
      kernel->setArg(idx++, gws[2]);
    }
    kernel->setArg(idx++,
                   *(static_cast<const cl::Image2D *>(input0->opencl_image())));
    kernel->setArg(idx++,
                   *(static_cast<const cl::Image2D *>(input1->opencl_image())));
    kernel->setArg(idx++, static_cast<int32_t>(input0->dim(3)));
    kernel->setArg(idx++,
                   *(static_cast<cl::Image2D *>(output->opencl_image())));

    *prev_input_shape = input0->shape();
  }

  const std::vector<uint32_t> lws = {8, *kwg_size / 64, 8, 1};
  std::stringstream ss;
  ss << "concat_opencl_kernel_" << output->dim(0) << "_" << output->dim(1)
     << "_" << output->dim(2) << "_" << output->dim(3);
  TuningOrRun3DKernel(*kernel, ss.str(), gws, lws, future);

  if (runtime->IsOutOfRangeCheckEnabled()) {
    (*kernel_error)->Map(nullptr);
    char *kerror_code = (*kernel_error)->mutable_data<char>();
    MACE_CHECK(*kerror_code == '0') << "Kernel error code: " << *kerror_code;
    (*kernel_error)->UnMap();
  }
}

static void ConcatN(cl::Kernel *kernel,
                    const std::vector<const Tensor *> &input_list,
                    const DataType dt,
                    Tensor *output,
                    StatsFuture *future,
                    uint32_t *kwg_size,
                    std::unique_ptr<BufferBase> *kernel_error) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channel = output->dim(3);

  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("concat_channel_multi");
    built_options.emplace("-Dconcat_channel_multi=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    if (runtime->IsOutOfRangeCheckEnabled()) {
      built_options.emplace("-DOUT_OF_RANGE_CHECK");
      *kernel_error = std::move(std::unique_ptr<Buffer>(
            new Buffer(GetDeviceAllocator(DeviceType::OPENCL), 1)));
      (*kernel_error)->Map(nullptr);
      *((*kernel_error)->mutable_data<char>()) = '0';
      (*kernel_error)->UnMap();
    }
    if (runtime->IsNonUniformWorkgroupsSupported()) {
      built_options.emplace("-DNON_UNIFORM_WORK_GROUP");
    }
    *kernel = runtime->BuildKernel("concat", kernel_name, built_options);
    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const int inputs_count = input_list.size();
  index_t chan_blk_offset = 0;
  for (int i = 0; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    index_t input_channel_blk = input->dim(3) / 4;
    const uint32_t gws[3] = {
        static_cast<uint32_t>(input_channel_blk), static_cast<uint32_t>(width),
        static_cast<uint32_t>(batch * height),
    };

    uint32_t idx = 0;
    if (runtime->IsOutOfRangeCheckEnabled()) {
      kernel->setArg(idx++,
          *(static_cast<cl::Buffer *>((*kernel_error)->buffer())));
    }
    if (!runtime->IsNonUniformWorkgroupsSupported()) {
      kernel->setArg(idx++, gws[0]);
      kernel->setArg(idx++, gws[1]);
      kernel->setArg(idx++, gws[2]);
    }
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, static_cast<int32_t>(chan_blk_offset));
    kernel->setArg(idx++, *(output->opencl_image()));

    chan_blk_offset += input_channel_blk;
    const std::vector<uint32_t> lws = {8, *kwg_size / 64, 8, 1};
    std::stringstream ss;
    ss << "concat_n_opencl_kernel_" << input_channel_blk << "_" << width << "_"
       << batch * height;
    TuningOrRun3DKernel(*kernel, ss.str(), gws, lws, future);

    if (runtime->IsOutOfRangeCheckEnabled()) {
      (*kernel_error)->Map(nullptr);
      char *kerror_code = (*kernel_error)->mutable_data<char>();
      MACE_CHECK(*kerror_code == '0') << "Kernel error code: " << *kerror_code;
      (*kernel_error)->UnMap();
    }
  }
}

template <typename T>
void ConcatFunctor<DeviceType::OPENCL, T>::operator()(
    const std::vector<const Tensor *> &input_list,
    Tensor *output,
    StatsFuture *future) {
  const int inputs_count = input_list.size();
  MACE_CHECK(inputs_count >= 2 && axis_ == 3)
      << "Concat opencl kernel only support >=2 elements with axis == 3";

  const Tensor *input0 = input_list[0];
  bool divisible_four = input0->dim(axis_) % 4 == 0;

  std::vector<index_t> output_shape(input0->shape());
  for (int i = 1; i < inputs_count; ++i) {
    const Tensor *input = input_list[i];
    MACE_CHECK(input->dim_size() == input0->dim_size(),
               "Ranks of all input tensors must be same.");
    divisible_four &= input->dim(axis_) % 4 == 0;
    for (int j = 0; j < input->dim_size(); ++j) {
      if (j == axis_) {
        continue;
      }
      MACE_CHECK(input->dim(j) == input0->dim(j),
                 "Dimensions of inputs should equal except axis.");
    }
    output_shape[axis_] += input->dim(axis_);
  }
  MACE_CHECK(
      inputs_count == 2 || divisible_four,
      "Dimensions of inputs should be divisible by 4 when inputs_count > 2.");
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  output->ResizeImage(output_shape, image_shape);

  switch (inputs_count) {
    case 2:
      Concat2(&kernel_, input_list[0], input_list[1], DataTypeToEnum<T>::value,
              &input_shape_, output, future, &kwg_size_, &kernel_error_);
      break;
    default:
      if (divisible_four) {
        ConcatN(&kernel_, input_list, DataTypeToEnum<T>::value, output, future,
            &kwg_size_, &kernel_error_);
      } else {
        MACE_NOT_IMPLEMENTED;
      }
  }
}

template struct ConcatFunctor<DeviceType::OPENCL, float>;
template struct ConcatFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
