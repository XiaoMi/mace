//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/depthwise_conv2d.h"
#include "mace/kernels/activation.h"

namespace mace {
namespace kernels {

namespace {

void DepthwiseConv2dNCHW(const float *input,
                         const float *filter,
                         const index_t batch,
                         const index_t in_height,
                         const index_t in_width,
                         const index_t in_channels,
                         const index_t out_height,
                         const index_t out_width,
                         const index_t out_channels,
                         const int filter_height,
                         const int filter_width,
                         const int stride_h,
                         const int stride_w,
                         const int dilation_h,
                         const int dilation_w,
                         const int pad_top,
                         const int pad_left,
                         float *output) {
  const index_t multiplier = out_channels / in_channels;
#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t m = 0; m < out_channels; ++m) {
      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          index_t out_offset =
            ((b * out_channels + m) * out_height + h) * out_width + w;
          index_t c = m / multiplier;
          index_t o = m % multiplier;
          float sum = 0;
          for (index_t kh = 0; kh < filter_height; ++kh) {
            for (index_t kw = 0; kw < filter_width; ++kw) {
              index_t ih = h * stride_h + kh * dilation_h - pad_top;
              index_t iw = w * stride_w + kw * dilation_w - pad_left;
              if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                index_t in_offset =
                  ((b * in_channels + c) * in_height + ih) * in_width + iw;
                index_t filter_offset =
                  (((o * in_channels) + c) * filter_height + kh) * filter_width
                    + kw;

                sum += input[in_offset] * filter[filter_offset];
              }
            }
          }
          output[out_offset] = sum;
        }
      }
    }
  }
}
}  // namespace

extern void DepthwiseConv2dNeonK3x3S1(const float *input,
                                      const float *filter,
                                      const index_t batch,
                                      const index_t in_height,
                                      const index_t in_width,
                                      const index_t in_channels,
                                      const index_t out_height,
                                      const index_t out_width,
                                      const index_t out_channels,
                                      const int pad_top,
                                      const int pad_left,
                                      const int valid_h_start,
                                      const int valid_h_stop,
                                      const int valid_w_start,
                                      const int valid_w_stop,
                                      float *output);

void DepthwiseConv2dNeonK3x3S2(const float *input,
                               const float *filter,
                               const index_t batch,
                               const index_t in_height,
                               const index_t in_width,
                               const index_t in_channels,
                               const index_t out_height,
                               const index_t out_width,
                               const index_t out_channels,
                               const int pad_top,
                               const int pad_left,
                               const int valid_h_start,
                               const int valid_h_stop,
                               const int valid_w_start,
                               const int valid_w_stop,
                               float *output);

void DepthwiseConv2dFunctor<DeviceType::NEON,
                            float>::operator()(const Tensor *input,
                                               const Tensor *filter,
                                               const Tensor *bias,
                                               Tensor *output,
                                               StatsFuture *future) {
  MACE_CHECK_NOTNULL(input);
  MACE_CHECK_NOTNULL(filter);
  MACE_CHECK_NOTNULL(output);

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  std::vector<index_t> filter_shape
    {filter->dim(0) * filter->dim(1), filter->dim(1), filter->dim(2),
     filter->dim(3)};

  if (paddings_.empty()) {
    CalcNCHWPaddingAndOutputSize(input->shape().data(),
                                 filter_shape.data(),
                                 dilations_,
                                 strides_,
                                 padding_type_,
                                 output_shape.data(),
                                 paddings.data());
  } else {
    paddings = paddings_;
    CalcNCHWOutputSize(input->shape().data(), filter_shape.data(),
                       paddings_.data(), dilations_, strides_, RoundType::FLOOR,
                       output_shape.data());
  }
  output->Resize(output_shape);
  output->Clear();

  index_t batch = output->dim(0);
  index_t channels = output->dim(1);
  index_t height = output->dim(2);
  index_t width = output->dim(3);

  index_t input_batch = input->dim(0);
  index_t input_channels = input->dim(1);
  index_t input_height = input->dim(2);
  index_t input_width = input->dim(3);

  index_t filter_h = filter_shape[2];
  index_t filter_w = filter_shape[3];
  MACE_CHECK(filter_shape[0] == channels, filter_shape[0], " != ", channels);
  MACE_CHECK(filter_shape[1] == input_channels, filter_shape[1], " != ",
             input_channels);

  index_t stride_h = strides_[0];
  index_t stride_w = strides_[1];

  index_t dilation_h = dilations_[0];
  index_t dilation_w = dilations_[1];

  MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

  int pad_top = paddings[0] >> 1;
  int pad_bottom = paddings[0] - pad_top;
  int pad_left = paddings[1] >> 1;
  int pad_right = paddings[1] - pad_left;

  int valid_h_start = pad_top == 0 ? 0 : (pad_top - 1) / stride_h + 1;
  int valid_h_stop = pad_bottom == 0
                     ? height
                     : height - ((pad_bottom - 1) / stride_h + 1);
  int valid_w_start = pad_left == 0 ? 0 : (pad_left - 1) / stride_w + 1;
  int valid_w_stop = pad_right == 0
                     ? width
                     : width - ((pad_right - 1) / stride_w + 1);

  std::function<void(const float *input, float *output)> conv_func;

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
  auto output_data = output->mutable_data<float>();

  if (filter_h == 3 && filter_w == 3 && stride_h == 1 && stride_w == 1
    && dilation_h == 1 && dilation_w == 1) {
    conv_func = [=](const float *input, float *output) {
      DepthwiseConv2dNeonK3x3S1(input,
                                filter_data,
                                batch,
                                input_height,
                                input_width,
                                input_channels,
                                height,
                                width,
                                channels,
                                pad_top,
                                pad_left,
                                valid_h_start,
                                valid_h_stop,
                                valid_w_start,
                                valid_w_stop,
                                output);
    };
  } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 && stride_w == 2
    && dilation_h == 1 && dilation_w == 1) {
    conv_func = [=](const float *input, float *output) {
      DepthwiseConv2dNeonK3x3S2(input,
                                filter_data,
                                batch,
                                input_height,
                                input_width,
                                input_channels,
                                height,
                                width,
                                channels,
                                pad_top,
                                pad_left,
                                valid_h_start,
                                valid_h_stop,
                                valid_w_start,
                                valid_w_stop,
                                output);
    };
  } else {
    conv_func = [=](const float *input, float *output) {
      DepthwiseConv2dNCHW(input,
                          filter_data,
                          batch,
                          input_height,
                          input_width,
                          input_channels,
                          height,
                          width,
                          channels,
                          filter_h,
                          filter_w,
                          stride_h,
                          stride_w,
                          dilation_h,
                          dilation_w,
                          pad_top,
                          pad_left,
                          output);
    };
  }

  conv_func(input_data, output_data);

  if (bias_data != nullptr) {
#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        for (index_t i = 0; i < height * width; ++i) {
          output_data[(b * channels + c) * height * width + i] += bias_data[c];
        }
      }
    }
  }

  DoActivation(output_data, output_data, output->size(), activation_,
               relux_max_limit_);
}

}  // namespace kernels
}  // namespace mace
