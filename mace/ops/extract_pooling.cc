// Copyright 2018 The MACE Authors. All Rights Reserved.
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

// This Op is for fused StatisticsExtraction, StatisticsPooling and
// Round Components in Kaldi.
// This op is used to extract moving-average mean and standard-deviation
// statistics of input data.
// 'input_indexes' indicates which frames will be used for extract statistics.
// 'output_indexes' indicates which frames  of outputs will be used to
// save statistics results.
// 'modulus' will be used for extent results to all frames.
// 'start_index' and 'end_index' indicate time indexes of output frames.
// 'forward_indexes' and 'count' were from precomputed index in kaldi.
// Reference to
// http://kaldi-asr.org/doc/nnet-general-component_8h_source.html#l00158

#include <functional>
#include <memory>

#include "mace/core/operator.h"


namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ExtractPoolingOp;

template <typename T>
class ExtractPoolingOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ExtractPoolingOp(OpConstructContext *context)
      : Operation(context),
        modulus_(Operation::GetOptionalArg<int>("modulus", 1)),
        include_variance_(
            static_cast<bool>(
                Operation::GetOptionalArg<int>("include_variance", 0))),
        num_log_count_(
            Operation::GetOptionalArg<int>("num_log_count", 0)),
        variance_floor_(
            Operation::GetOptionalArg<float>("variance_floor", 1.0e-10)),
        input_indexes_(Operation::GetRepeatedArgs<int>("input_indexes")),
        output_indexes_(Operation::GetRepeatedArgs<int>("output_indexes")),
        forward_indexes_(Operation::GetRepeatedArgs<int>("forward_indexes")),
        counts_(Operation::GetRepeatedArgs<float>("counts")),
        input_time_range_(Operation::GetRepeatedArgs<int>("input_time_range")),
        output_time_range_(
            Operation::GetRepeatedArgs<int>("output_time_range")) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    const std::vector<index_t> &input_shape = input->shape();
    const index_t dim_size = input_shape.size();
    MACE_CHECK(dim_size >= 2,
               "ExtractPooling only supports input dim size >= 2");
    MACE_CHECK(modulus_ >= 1,
               "ExtractPooling's pooling size should be greater than zero.");
    MACE_CHECK(input_time_range_.size() == 2 && output_time_range_.size() == 2
                   && counts_.size() * 2 == forward_indexes_.size()
                   && counts_.size() == output_indexes_.size());
    int in_start_index = input_time_range_[0];
    int out_start_index = output_time_range_[0];
    int out_end_index = output_time_range_[1];
    MACE_CHECK(out_end_index >= out_start_index
                   && input_time_range_[1] >= input_time_range_[0],
               "end index should be greater than start index.");
    const index_t output_chunk = out_end_index - out_start_index + 1;
    const index_t input_dim = input_shape[dim_size - 1];
    const index_t chunk = input_shape[dim_size - 2];
    MACE_CHECK(chunk == input_time_range_[1] - input_time_range_[0] + 1,
               "input chunk should be equal to end - start + 1.");
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());

    index_t output_dim = include_variance_ ? 2 * input_dim : input_dim;
    output_dim += num_log_count_;
    std::vector<index_t> output_shape(input_shape);
    output_shape[dim_size - 1] = output_dim;
    output_shape[dim_size - 2] = output_chunk;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    const index_t num_input_indexes = input_indexes_.size();
    const index_t num_output_indexes = output_indexes_.size();
    MACE_CHECK(num_input_indexes > 0 && num_output_indexes > 0,
               "ExtractPooling's input_indexes or output_indexes is empty.");
    const index_t extract_out_size = PadAlignSize(output_dim * sizeof(float));
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(extract_out_size);

    Tensor extract_out(scratch->Scratch(extract_out_size), DT_FLOAT);
    extract_out.Reshape({1, output_dim});
    extract_out.Clear();
    float *extract_out_data = extract_out.mutable_data<float>();

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    for (index_t b = 0; b < batch; ++b) {
      for (index_t i = 0; i < num_output_indexes; ++i) {
        int start = forward_indexes_[2 * i];
        int end = forward_indexes_[2 * i + 1];
        float count = counts_[i];
        float mean_scale = 1.f / count;
        float log_count = std::log(count);
        thread_pool.Compute1D([=](index_t start0,
                                  index_t end0,
                                  index_t step0) {
          for (index_t n = start0; n < end0; n += step0) {
            extract_out_data[n] = log_count;
          }
        }, 0, num_log_count_, 1);
        if (include_variance_) {
          thread_pool.Compute1D([=](index_t start0,
                                    index_t end0,
                                    index_t step0) {
            for (index_t d = start0; d < end0; d += step0) {
              float mean = 0.f;
              float variance = 0.f;
              for (int t = start; t < end; ++t) {
                index_t input_index =
                    (b * chunk + input_indexes_[t] - in_start_index)
                        * input_dim;
                float x = input_data[input_index + d];
                mean += x;
                variance += x * x;
              }
              mean *= mean_scale;
              variance *= mean_scale;
              extract_out_data[d + num_log_count_] = mean;
              variance = variance - mean * mean;
              extract_out_data[d + input_dim + num_log_count_] =
                  variance < variance_floor_ ?
                  std::sqrt(variance_floor_) :
                  std::sqrt(variance);
            }
          }, 0, input_dim, 1);
        } else {
          thread_pool.Compute1D([=](index_t start0,
                                    index_t end0,
                                    index_t step0) {
            for (index_t d = start0; d < end0; d += step0) {
              float mean = 0.f;
              for (int t = start; t < end; ++t) {
                index_t input_index =
                    (b * chunk + input_indexes_[t] - in_start_index)
                        * input_dim;
                mean += input_data[input_index + d];
              }
              extract_out_data[d + num_log_count_] = mean * mean_scale;
            }
          }, 0, input_dim, 1);
        }

        int output_start = output_indexes_[i] < out_start_index ?
                           out_start_index : output_indexes_[i];
        int output_end = output_indexes_[i] + modulus_;
        output_end = output_end > out_end_index ?
                     out_end_index + 1 :
                     output_end;
        thread_pool.Compute1D([=](index_t start0,
                                  index_t end0,
                                  index_t step0) {
          for (index_t idx = start0; idx < end0; idx += step0) {
            memcpy(output_data + (b * output_chunk + idx - out_start_index)
                       * output_dim,
                   extract_out_data, output_dim * sizeof(float));
          }
        }, output_start, output_end, 1);
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int modulus_;
  bool include_variance_;
  int num_log_count_;
  float variance_floor_;
  std::vector<int> input_indexes_;
  std::vector<int> output_indexes_;
  std::vector<int> forward_indexes_;
  std::vector<float> counts_;
  std::vector<int> input_time_range_;
  std::vector<int> output_time_range_;
};

void RegisterExtractPooling(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ExtractPooling", ExtractPoolingOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
