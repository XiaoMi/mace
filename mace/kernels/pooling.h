//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_POOLING_H
#define MACE_KERNELS_POOLING_H

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };
enum { PADDING_DROP = 0, PADDING_ZERO = 1};

template<typename T>
void PoolingFunction(const Tensor *input_tensor, Tensor *output_tensor, int pooling_type,
                     int kernel_size, int stride, int padding) {
  // max value in NxN window
  // avg value in NxN window

  vector <int64> in_shape = input_tensor->shape();
  REQUIRE(in_shape.size() == 4, "The input tensor shape should specify 4 dimensions(NCHW)");

  int64 batch_size = in_shape[0];
  int64 channels = in_shape[1];
  int64 h = in_shape[2];
  int64 w = in_shape[3];

  // calculate paddings and output tensor shape
  int outw, outh, pad_top, pad_bottom, pad_left, pad_right;
  if (padding == PADDING_ZERO) {
    int wpad = kernel_size + (w - 1) / stride * stride - w;
    int hpad = kernel_size + (h - 1) / stride * stride - h;

    pad_top = hpad / 2;
    pad_bottom = hpad - pad_top;
    pad_left = wpad / 2;
    pad_right = wpad - pad_left;

    outw = (w + wpad - kernel_size) / stride + 1;
    outh = (h + hpad - kernel_size) / stride + 1;
  } else if (padding == PADDING_DROP)  // Drop bottom-most rows and right-most columns
  {
    pad_top = pad_bottom = pad_left = pad_right = 0;

    outw = (w - kernel_size) / stride + 1;
    outh = (h - kernel_size) / stride + 1;
  }

  output_tensor->Resize({batch_size, channels, outh, outw});

  if (pooling_type == PoolMethod_MAX) {
#pragma omp parallel for
    for (int batch = 0; batch < batch_size; batch++) {
      for (int q = 0; q < channels; q++) {
        float *outptr = output_tensor->mutable_data<float>() + (batch * channels + q) * outw * outh;

        for (int i = 0; i < outh; i++) {
          for (int j = 0; j < outw; j++) {
            float val;
            float max;
            if (padding == PADDING_ZERO) {
              max = 0.0;
              for (int m = 0; m < kernel_size; m++) {
                for (int n = 0; n < kernel_size; n++) {
                  if (i * stride - pad_top + m < 0 || j * stride - pad_left + n < 0 ||
                      i * stride - pad_top + m >= h || j * stride - pad_left + n >= w) {
                    val = 0.0;
                  } else {
                    int index = (batch * channels + q) * w * h + w * (i * stride - pad_top + m) + j * stride - pad_left + n;
                    val = input_tensor->data<float>()[index];
                  }
                  max = std::max(max, val);
                }
              }
            } else {
              const float *sptr = input_tensor->data<float>() + (batch * channels + q) * w * h + w * i * stride + j * stride;
              max = sptr[0];
              for (int m = 0; m < kernel_size; m++) {
                for (int n = 0; n < kernel_size; n++) {
                  val = sptr[w * m + n];
                  max = std::max(max, val);
                }
              }
            }
            outptr[j] = max;
          }

          outptr += outw;
        }
      }
    }
  } else if (pooling_type == PoolMethod_AVE) {
#pragma omp parallel for
    for (int batch = 0; batch < batch_size; batch++) {
      for (int q = 0; q < channels; q++) {
        float *outptr = output_tensor->mutable_data<float>() + (batch * channels + q) * outw * outh;

        for (int i = 0; i < outh; i++) {
          for (int j = 0; j < outw; j++) {
            float val;
            float sum = 0.0;
            if (padding == PADDING_ZERO) {
              for (int m = 0; m < kernel_size; m++) {
                for (int n = 0; n < kernel_size; n++) {
                  if (i * stride - pad_top + m < 0 || j * stride - pad_left + n < 0 ||
                      i * stride - pad_top + m >= h || j * stride - pad_left + n >= w) {
                    val = 0.0;
                  } else {
                    int index =
                    (batch * channels + q) * w * h + w * (i * stride - pad_top + m) + j * stride - pad_left + n;
                    val = input_tensor->data<float>()[index];
                  }
                  sum += val;
                }
              }
            } else {
              const float *sptr =
              input_tensor->data<float>() + (batch * channels + q) * w * h + w * i * stride + j * stride;
              for (int m = 0; m < kernel_size; m++) {
                for (int n = 0; n < kernel_size; n++) {
                  val = sptr[w * m + n];
                  sum += val;
                }
              }
            }
            outptr[j] = sum / (kernel_size * kernel_size);
          }

          outptr += outw;
        }
      }
    }
  }

}
} //  namespace kernels
} //  namespace mace

#endif //MACE_KERNELS_POOLING_H
