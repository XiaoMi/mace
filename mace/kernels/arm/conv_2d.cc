//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/kernels/arm/conv_winograd.h"

// winograd is always superior to neon impl during benchmark
#define USE_WINOGRAD 1

namespace mace {
namespace kernels {

namespace {

void Conv2dNCHW(const float *input,
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
                float *output) {
#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t m = 0; m < out_channels; ++m) {
      for (index_t h = 0; h < out_height; ++h) {
        for (index_t w = 0; w < out_width; ++w) {
          index_t out_offset =
            ((b * out_channels + m) * out_height + h) * out_width + w;
          for (index_t c = 0; c < in_channels; ++c) {
            for (index_t kh = 0; kh < filter_height; ++kh) {
              for (index_t kw = 0; kw < filter_width; ++kw) {
                index_t ih = h * stride_h + kh * dilation_h;
                index_t iw = w * stride_w + kw * dilation_w;
                index_t in_offset =
                  ((b * in_channels + c) * in_height + ih) * in_width + iw;
                index_t filter_offset =
                  (((m * in_channels) + c) * filter_height + kh) * filter_width
                    + kw;
                output[out_offset] += input[in_offset] * filter[filter_offset];
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

extern void Conv2dNeonK1x1S1(const float *input,
                             const float *filter,
                             const index_t batch,
                             const index_t height,
                             const index_t width,
                             const index_t in_channels,
                             const index_t out_channels,
                             float *output);

extern void Conv2dNeonK3x3S1(const float *input,
                             const float *filter,
                             const index_t batch,
                             const index_t in_height,
                             const index_t in_width,
                             const index_t in_channels,
                             const index_t out_height,
                             const index_t out_width,
                             const index_t out_channels,
                             float *output);

extern void Conv2dNeonK3x3S2(const float *input,
                             const float *filter,
                             const index_t batch,
                             const index_t in_height,
                             const index_t in_width,
                             const index_t in_channels,
                             const index_t out_height,
                             const index_t out_width,
                             const index_t out_channels,
                             float *output);

void Conv2dFunctor<DeviceType::NEON, float>::operator()(const Tensor *input,
                                                        const Tensor *filter,
                                                        const Tensor *bias,
                                                        Tensor *output,
                                                        StatsFuture *future) {
  MACE_CHECK_NOTNULL(input);
  MACE_CHECK_NOTNULL(filter);
  MACE_CHECK_NOTNULL(output);

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (paddings_.empty()) {
    CalcNCHWPaddingAndOutputSize(input->shape().data(),
                                 filter->shape().data(),
                                 dilations_,
                                 strides_,
                                 padding_type_,
                                 output_shape.data(),
                                 paddings.data());
  } else {
    paddings = paddings_;
    CalcNCHWOutputSize(input->shape().data(), filter->shape().data(),
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

  index_t filter_h = filter->dim(2);
  index_t filter_w = filter->dim(3);
  MACE_CHECK(filter->dim(0) == channels, filter->dim(0), " != ", channels);
  MACE_CHECK(filter->dim(1) == input_channels, filter->dim(1), " != ",
             input_channels);

  index_t stride_h = strides_[0];
  index_t stride_w = strides_[1];

  index_t dilation_h = dilations_[0];
  index_t dilation_w = dilations_[1];

  MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

  index_t padded_input_height = input_height + paddings[0];
  index_t padded_input_width = input_width + paddings[1];
  index_t extra_input_height = padded_input_height;
  index_t extra_input_width = padded_input_width;
  index_t extra_output_height = height;
  index_t extra_output_width = width;

  int pad_top = paddings[0] >> 1;
  int pad_bottom = paddings[0] - pad_top;
  int pad_left = paddings[1] >> 1;
  int pad_right = paddings[1] - pad_left;

  std::function<void(const float *input, float *output)> conv_func;

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
  auto output_data = output->mutable_data<float>();

  if (USE_WINOGRAD && filter_h == 3 && filter_w == 3 && stride_h == 1
    && stride_w == 1
    && dilation_h == 1 && dilation_w == 1) {
    extra_output_height = RoundUp<index_t>(height, 2);
    extra_input_height = std::max(padded_input_height, extra_output_height + 2);
    extra_output_width = RoundUp<index_t>(width, 2);
    extra_input_width = std::max(padded_input_width, extra_output_width + 2);
    if (extra_input_height != padded_input_height) {
      pad_bottom += (extra_input_height - padded_input_height);
    }
    if (extra_input_width != padded_input_width) {
      pad_right += (extra_input_width - padded_input_width);
    }

    index_t tile_height_count = (extra_output_height + 1) / 2;
    index_t tile_width_count = (extra_output_width + 1) / 2;
    index_t tile_count = tile_height_count * tile_width_count;
    transformed_input_.Resize({16, batch, input_channels, tile_count});
    transformed_filter_.Resize({16, channels, input_channels});
    transformed_output_.Resize({16, batch, channels, tile_count});

    conv_func = [=](const float *pad_input, float *pad_output) {
      WinoGradConv3x3s1(pad_input,
                        filter_data,
                        batch,
                        extra_input_height,
                        extra_input_width,
                        input_channels,
                        channels,
                        transformed_input_.mutable_data<float>(),
                        transformed_filter_.mutable_data<float>(),
                        transformed_output_.mutable_data<float>(),
                        is_filter_transformed_,
                        pad_output);
      is_filter_transformed_ = true;
    };
  } else if (filter_h == 3 && filter_w == 3 && stride_h == 1 && stride_w == 1
    && dilation_h == 1 && dilation_w == 1) {
    extra_output_height = RoundUp<index_t>(height, 2);
    extra_input_height = std::max(padded_input_height, extra_output_height + 2);
    extra_output_width = RoundUp<index_t>(width, 4);
    extra_input_width = std::max(padded_input_width, extra_output_width + 2);
    if (extra_input_height != padded_input_height) {
      pad_bottom += (extra_input_height - padded_input_height);
    }
    if (extra_input_width != padded_input_width) {
      pad_right += (extra_input_width - padded_input_width);
    }

    conv_func = [=](const float *pad_input, float *pad_output) {
      Conv2dNeonK3x3S1(pad_input,
                       filter_data,
                       batch,
                       extra_input_height,
                       extra_input_width,
                       input_channels,
                       extra_output_height,
                       extra_output_width,
                       channels,
                       pad_output);
    };
  } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 && stride_w == 2
    && dilation_h == 1 && dilation_w == 1) {
    extra_output_height = height;
    extra_input_height =
      std::max(padded_input_height, (extra_output_height - 1) * 2 + 3);
    extra_output_width = RoundUp<index_t>(width, 4);
    extra_input_width =
      std::max(padded_input_width, (extra_output_width - 1) * 2 + 3);
    if (extra_input_height != padded_input_height) {
      pad_bottom += (extra_input_height - padded_input_height);
    }
    if (extra_input_width != padded_input_width) {
      pad_right += (extra_input_width - padded_input_width);
    }

    conv_func = [=](const float *pad_input, float *pad_output) {
      Conv2dNeonK3x3S2(pad_input,
                       filter_data,
                       batch,
                       extra_input_height,
                       extra_input_width,
                       input_channels,
                       extra_output_height,
                       extra_output_width,
                       channels,
                       pad_output);
    };
  } else if (filter_h == 1 && filter_w == 1 && stride_h == 1 && stride_w == 1
    && dilation_h == 1 && dilation_w == 1) {
    conv_func = [=](const float *pad_input, float *pad_output) {
      Conv2dNeonK1x1S1(input_data,
                       filter_data,
                       batch,
                       height,
                       width,
                       input_channels,
                       channels,
                       output_data);
    };
  } else {
    conv_func = [=](const float *pad_input, float *pad_output) {
      Conv2dNCHW(pad_input,
                 filter_data,
                 batch,
                 extra_input_height,
                 extra_input_width,
                 input_channels,
                 extra_output_height,
                 extra_output_width,
                 channels,
                 filter_h,
                 filter_w,
                 stride_h,
                 stride_w,
                 dilation_h,
                 dilation_w,
                 pad_output);
    };
  }

  const Tensor *pad_input_ptr = input;
  // Keep this alive during kernel execution
  if (extra_input_height != input_height || extra_input_width != input_width) {
    ConstructNCHWInputWithSpecificPadding(input,
                                          pad_top,
                                          pad_bottom,
                                          pad_left,
                                          pad_right,
                                          &padded_input_);
    pad_input_ptr = &padded_input_;
  }
  const float *pad_input_data = pad_input_ptr->data<float>();

  Tensor *pad_output_ptr = output;
  // Keep this alive during kernel execution
  if (extra_output_height != height || extra_output_width != width) {
    std::vector<index_t> extra_output_shape
      {batch, channels, extra_output_height, extra_output_width};
    padded_output_.Resize(extra_output_shape);
    padded_output_.Clear();
    pad_output_ptr = &padded_output_;
  }
  float *pad_output_data = pad_output_ptr->mutable_data<float>();

  conv_func(pad_input_data, pad_output_data);

  // unpack output
  if (extra_output_height != height || extra_output_width != width) {
#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        for (index_t h = 0; h < height; ++h) {
          memcpy(
            output_data + b * channels * height * width + c * height * width
              + h * width,
            pad_output_data
              + b * channels * extra_output_height * extra_output_width
              + c * extra_output_height * extra_output_width
              + h * extra_output_width,
            sizeof(float) * width);
        }
      }
    }
  }

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
