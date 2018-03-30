//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/pooling.h"

namespace mace {
namespace kernels {

namespace {

void MaxPooling(const float *input,
                const index_t batch,
                const index_t in_height,
                const index_t in_width,
                const index_t channels,
                const index_t out_height,
                const index_t out_width,
                const int filter_height,
                const int filter_width,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int pad_top,
                const int pad_left,
                float *output) {
  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = channels * in_image_size;
  const index_t out_batch_size = channels * out_image_size;

#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t out_base = b * out_batch_size + c * out_image_size;
      const index_t in_base = b * in_batch_size + c * in_image_size;
      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          const index_t out_offset = out_base + h * out_width + w;
          float res = std::numeric_limits<float>::lowest();
          for (int fh = 0; fh < filter_height; ++fh) {
            for (int fw = 0; fw < filter_width; ++fw) {
              int inh = h * stride_h + dilation_h * fh - pad_top;
              int inw = w * stride_w + dilation_w * fw - pad_left;
              if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                index_t input_offset = in_base + inh * in_width + inw;
                res = std::max(res, input[input_offset]);
              }
            }
          }
          output[out_offset] = res;
        }
      }
    }
  }
}

void AvgPooling(const float *input,
                const index_t batch,
                const index_t in_height,
                const index_t in_width,
                const index_t channels,
                const index_t out_height,
                const index_t out_width,
                const int filter_height,
                const int filter_width,
                const int stride_h,
                const int stride_w,
                const int dilation_h,
                const int dilation_w,
                const int pad_top,
                const int pad_left,
                float *output) {
  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = channels * in_image_size;
  const index_t out_batch_size = channels * out_image_size;

#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t out_base = b * out_batch_size + c * out_image_size;
      const index_t in_base = b * in_batch_size + c * in_image_size;
      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          const index_t out_offset = out_base + h * out_width + w;
          float res = 0;
          int block_size = 0;
          for (int fh = 0; fh < filter_height; ++fh) {
            for (int fw = 0; fw < filter_width; ++fw) {
              int inh = h * stride_h + dilation_h * fh - pad_top;
              int inw = w * stride_w + dilation_w * fw - pad_left;
              if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                index_t input_offset = in_base + inh * in_width + inw;
                res += input[input_offset];
                ++block_size;
              }
            }
          }
          output[out_offset] = res / block_size;
        }
      }
    }
  }
}
}  // namespace

void PoolingFunctor<DeviceType::NEON,
                    float>::operator()(const Tensor *input_tensor,
                                       Tensor *output_tensor,
                                       StatsFuture *future) {
  std::vector<index_t> output_shape(4);
  std::vector<index_t> filter_shape = {
    input_tensor->dim(1), input_tensor->dim(1), kernels_[0], kernels_[1]};

  std::vector<int> paddings(2);
  if (paddings_.empty()) {
    kernels::CalcNCHWPaddingAndOutputSize(
      input_tensor->shape().data(), filter_shape.data(), dilations_,
      strides_, padding_type_, output_shape.data(), paddings.data());
  } else {
    paddings = paddings_;
    CalcNCHWOutputSize(input_tensor->shape().data(), filter_shape.data(),
                       paddings_.data(), dilations_, strides_, RoundType::CEIL,
                       output_shape.data());
  }
  output_tensor->Resize(output_shape);

  const float *input = input_tensor->data<float>();
  float *output = output_tensor->mutable_data<float>();
  const index_t *input_shape = input_tensor->shape().data();
  index_t batch = output_shape[0];
  index_t channels = output_shape[1];
  index_t height = output_shape[2];
  index_t width = output_shape[3];

  index_t input_channels = input_shape[1];
  index_t input_height = input_shape[2];
  index_t input_width = input_shape[3];

  index_t in_image_size = input_height * input_width;

  int filter_h = kernels_[0];
  int filter_w = kernels_[1];

  int stride_h = strides_[0];
  int stride_w = strides_[1];

  int dilation_h = dilations_[0];
  int dilation_w = dilations_[1];

  int pad_top = paddings[0] / 2;
  int pad_left = paddings[1] / 2;

  if (pooling_type_ == PoolingType::MAX) {
    MaxPooling(input,
               batch,
               input_height,
               input_width,
               channels,
               height,
               width,
               filter_h,
               filter_w,
               stride_h,
               stride_w,
               dilation_h,
               dilation_w,
               pad_top,
               pad_left,
               output);
  } else if (pooling_type_ == PoolingType::AVG) {
    AvgPooling(input,
               batch,
               input_height,
               input_width,
               channels,
               height,
               width,
               filter_h,
               filter_w,
               stride_h,
               stride_w,
               dilation_h,
               dilation_w,
               pad_top,
               pad_left,
               output);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

}  // namespace kernels
}  // namespace mace
