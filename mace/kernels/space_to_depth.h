//
// Created by liutuo on 18-3-20.
//

#ifndef MACE_KERNELS_SPACE_TO_DEPTH_H
#define MACE_KERNELS_SPACE_TO_DEPTH_H

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct SpaceToDepthOpFunctor {
  explicit SpaceToDepthOpFunctor(const int block_size) : block_size_(block_size) {}
  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    
    const int batch_size = input->dim(0);
    const int input_height = input->dim(1);
    const int input_width = input->dim(2);
    const int input_depth = input->dim(3);
    
    const index_t output_depth = input_depth * block_size_ * block_size_;
    const index_t output_width = input_width / block_size_;
    const index_t output_height = input_height / block_size_;
    
    std::vector<index_t> output_shape = {batch_size, output_height, output_width, output_depth};
    
    output->Resize(output_shape);

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
	  for (int h = 0; h < input_height; ++h) {
		  const int out_h = h / block_size_;
		  const int offset_h = (h % block_size_);
		  for (int w = 0; w < input_width; ++w) {
			  const int out_w = w/ block_size_;
			  const int offset_w = (w % block_size_);
			  const int offset_d = (offset_h * block_size_ + offset_w) * input_depth;
			  for (int d = 0; d < input_depth; ++d) {
				  const int out_d = d + offset_d;
				  const int o_index = ((b * output_height + out_h) * output_width + out_w) * output_depth + out_d;
			      const int i_index = ((b * input_height + h) * input_width + w) * input_depth + d;
				  output_ptr[o_index] = input_ptr[i_index];
			  }
		  }
	  }
  }

  }
  const int block_size_;
};

template <typename T>
struct SpaceToDepthOpFunctor<DeviceType::OPENCL, T> {

  SpaceToDepthOpFunctor(const int block_size) : block_size_(block_size) {}
  void operator()(const Tensor *input, Tensor *output, StatsFuture *future);

  cl::Kernel kernel_;
  const int block_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif //MACE_KERNELS_SPACE_TO_DEPTH_H
