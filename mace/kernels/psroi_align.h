// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_KERNELS_PSROI_ALIGN_H_
#define MACE_KERNELS_PSROI_ALIGN_H_

#include <algorithm>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct PSROIAlignFunctor {
  PSROIAlignFunctor(const T spatial_scale,
                    const int output_dim,
                    const int group_size) :
      spatial_scale_(spatial_scale),
      output_dim_(output_dim),
      group_size_(group_size) {}

  MaceStatus operator()(const Tensor *input,
                  const Tensor *rois,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const int height = static_cast<int>(input->dim(1));
    const int width = static_cast<int>(input->dim(2));
    const int channels = static_cast<int>(input->dim(3));
    const int pooled_height = group_size_;
    const int pooled_width = group_size_;
    const T *input_ptr = input->data<T>();
    const T *rois_ptr = rois->data<T>();
    // Number of ROIs
    const index_t num_rois = rois->dim(0);
    const index_t batch_size = input->dim(0);

    MACE_RETURN_IF_ERROR(output->Resize({num_rois, pooled_height, pooled_width,
                                        output_dim_}));
    T *output_ptr = output->mutable_data<T>();

    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = rois_ptr[0];
      T roi_start_w =
          static_cast<T>(rois_ptr[1]) * spatial_scale_;
      T roi_start_h =
          static_cast<T>(rois_ptr[2]) * spatial_scale_;
      T roi_end_w =
          static_cast<T>(rois_ptr[3] + 1.) * spatial_scale_;
      T roi_end_h =
          static_cast<T>(rois_ptr[4] + 1.) * spatial_scale_;
      MACE_CHECK(roi_batch_ind >= 0);
      MACE_CHECK(roi_batch_ind < batch_size);

      // Force too small ROIs to be 1x1
      T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(0.1));
      T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.1));

      // Compute w and h at bottom
      T bin_size_h = roi_height / static_cast<T>(pooled_height);
      T bin_size_w = roi_width / static_cast<T>(pooled_width);

      const T *batch_data = input_ptr +
          roi_batch_ind * height * width * channels;

      std::vector<T> vhstart, vwstart, vhend, vwend;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          T hstart = static_cast<T>(ph) * bin_size_h
              + roi_start_h;
          T wstart = static_cast<T>(pw) * bin_size_w
              + roi_start_w;
          T hend = static_cast<T>(ph + 1) * bin_size_h
              + roi_start_h;
          T wend = static_cast<T>(pw + 1) * bin_size_w
              + roi_start_w;
          // Add roi offsets and clip to input boundaries
          hstart = std::min(std::max(hstart, static_cast<T>(0.)),
                            static_cast<T>(height));
          hend = std::min(std::max(hend, static_cast<T>(0.)),
                          static_cast<T>(height));
          wstart = std::min(std::max(wstart, static_cast<T>(0.)),
                            static_cast<T>(width));
          wend = std::min(std::max(wend, static_cast<T>(0.)),
                          static_cast<T>(width));

          vhstart.push_back(hstart);
          vwstart.push_back(wstart);
          vhend.push_back(hend);
          vwend.push_back(wend);
        }
      }

#pragma omp parallel for collapse(3)
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          for (int c = 0; c < output_dim_; ++c) {
            const int pool_index = ph * pooled_width + pw;
            const int out_idx = pool_index * output_dim_ + c;
            const int in_chan_idx = (c * pooled_height + ph)
                * pooled_width + pw;
            T hstart = vhstart[pool_index];
            T hend = vhend[pool_index];
            T wstart = vwstart[pool_index];
            T wend = vwend[pool_index];
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            T out_sum = 0;
            for (T h = hstart; h < hend; h += 1.) {
              for (T w = wstart; w < wend; w += 1.) {
                // Selecting four regular locations for bilinear interpolation
                int x_left = std::floor(w);
                int x_right = std::ceil(w);
                int y_bottom = std::floor(h);
                int y_top = std::ceil(h);

                int top_left_index = (y_top * width + x_left)
                    * channels + in_chan_idx;
                int top_right_index = (y_top * width + x_right)
                    * channels + in_chan_idx;
                int bottom_left_index = (y_bottom * width + x_left)
                    * channels + in_chan_idx;
                int bottom_right_index = (y_bottom * width + x_right)
                    * channels + in_chan_idx;

                bool is_top_left_in = x_left >= 0 && x_left <= width - 1
                    && y_top >= 0 && y_top <= height - 1;
                bool is_top_right_in = x_right >= 0 && x_right <= width - 1
                    && y_top >= 0 && y_top <= height - 1;
                bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
                    && y_bottom >= 0 && y_bottom <= height - 1;
                bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
                    && y_bottom >= 0 && y_bottom <= height - 1;

                if (is_top_left_in) {
                  out_sum += (1 - w + x_left) * (1 - y_top + h)
                      * batch_data[top_left_index];
                }
                if (is_top_right_in) {
                  out_sum += (1 - x_right + w) * (1 - y_top + h)
                      * batch_data[top_right_index];
                }
                if (is_bottom_left_in) {
                  out_sum += (1 - w + x_left) * (1 - h + y_bottom)
                      * batch_data[bottom_left_index];
                }
                if (is_bottom_right_in) {
                  out_sum += (1 - x_right + w) * (1 - h + y_bottom)
                      * batch_data[bottom_right_index];
                }
              }
            }

            T bin_area = (hend - hstart) * (wend - wstart);
            output_ptr[out_idx] = is_empty ? 0. : out_sum / bin_area;
          }
        }
      }

      // Increment ROI data pointer
      rois_ptr += 5;
      output_ptr += pooled_height * pooled_width * output_dim_;
    }

    return MACE_SUCCESS;
  }

  const T spatial_scale_;
  const int output_dim_;
  const int group_size_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_PSROI_ALIGN_H_
