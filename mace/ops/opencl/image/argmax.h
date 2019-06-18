#ifndef MACE_OPS_OPENCL_IMAGE_ARGMAX_H_
#define MACE_OPS_OPENCL_IMAGE_ARGMAX_H_

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class ArgMaxKernel {
  public:
  MaceStatus Compute(OpContext *context, const Tensor *input, Tensor *output);
  
  private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
  std::string tuning_key_prefix_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_ARGMAX_H_