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

#ifndef MACE_OPS_OPS_TEST_UTIL_H_
#define MACE_OPS_OPS_TEST_UTIL_H_

#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mace/core/net.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"
#include "mace/kernels/opencl/common.h"
#include "mace/ops/ops_register.h"
#include "mace/utils/utils.h"

namespace mace {
namespace ops {
namespace test {

class OpDefBuilder {
 public:
  OpDefBuilder(const char *type, const std::string &name) {
    op_def_.set_type(type);
    op_def_.set_name(name);
  }

  OpDefBuilder &Input(const std::string &input_name) {
    op_def_.add_input(input_name);
    return *this;
  }

  OpDefBuilder &Output(const std::string &output_name) {
    op_def_.add_output(output_name);
    return *this;
  }

  OpDefBuilder &OutputType(const std::vector<DataType> &output_type) {
    for (auto out_t : output_type) {
      op_def_.add_output_type(out_t);
    }
    return *this;
  }

  OpDefBuilder AddIntArg(const std::string &name, const int value) {
    auto arg = op_def_.add_arg();
    arg->set_name(name);
    arg->set_i(value);
    return *this;
  }

  OpDefBuilder AddFloatArg(const std::string &name, const float value) {
    auto arg = op_def_.add_arg();
    arg->set_name(name);
    arg->set_f(value);
    return *this;
  }

  OpDefBuilder AddStringArg(const std::string &name, const char *value) {
    auto arg = op_def_.add_arg();
    arg->set_name(name);
    arg->set_s(value);
    return *this;
  }

  OpDefBuilder AddIntsArg(const std::string &name,
                          const std::vector<int> &values) {
    auto arg = op_def_.add_arg();
    arg->set_name(name);
    for (auto value : values) {
      arg->add_ints(value);
    }
    return *this;
  }

  OpDefBuilder AddFloatsArg(const std::string &name,
                            const std::vector<float> &values) {
    auto arg = op_def_.add_arg();
    arg->set_name(name);
    for (auto value : values) {
      arg->add_floats(value);
    }
    return *this;
  }

  void Finalize(OperatorDef *op_def) const {
    MACE_CHECK(op_def != nullptr, "input should not be null.");
    *op_def = op_def_;
  }

  OperatorDef op_def_;
};

class OpsTestNet {
 public:
  OpsTestNet() : op_registry_(new OperatorRegistry()) {}

  template <DeviceType D, typename T>
  void AddInputFromArray(const std::string &name,
                         const std::vector<index_t> &shape,
                         const std::vector<T> &data,
                         const float scale = 0.0,
                         const int32_t zero_point = 0) {
    Tensor *input =
        ws_.CreateTensor(name, GetDeviceAllocator(D), DataTypeToEnum<T>::v());
    input->Resize(shape);
    Tensor::MappingGuard input_mapper(input);
    T *input_data = input->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(input->size()) == data.size());
    memcpy(input_data, data.data(), data.size() * sizeof(T));
    input->SetScale(scale);
    input->SetZeroPoint(zero_point);
  }

  template <DeviceType D, typename T>
  void AddRepeatedInput(const std::string &name,
                        const std::vector<index_t> &shape,
                        const T data) {
    Tensor *input =
        ws_.CreateTensor(name, GetDeviceAllocator(D), DataTypeToEnum<T>::v());
    input->Resize(shape);
    Tensor::MappingGuard input_mapper(input);
    T *input_data = input->mutable_data<T>();
    std::fill(input_data, input_data + input->size(), data);
  }

  template <DeviceType D, typename T>
  void AddRandomInput(const std::string &name,
                      const std::vector<index_t> &shape,
                      bool positive = true,
                      bool truncate = false) {
    Tensor *input =
        ws_.CreateTensor(name, GetDeviceAllocator(D), DataTypeToEnum<T>::v());
    input->Resize(shape);
    Tensor::MappingGuard input_mapper(input);
    T *input_data = input->mutable_data<T>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> nd(0, 1);
    if (DataTypeToEnum<T>::value == DT_HALF) {
      std::generate(
          input_data, input_data + input->size(),
          [&gen, &nd, positive, truncate] {
            float d = nd(gen);
            if (truncate) {
              if (std::abs(d) > 100.f) d = 100.f;
              if (std::abs(d) < 0.001f) d = 0.001f;
            }
            return half_float::half_cast<half>(positive ? std::abs(d) : d);
          });
    } else {
      std::generate(input_data, input_data + input->size(),
                    [&gen, &nd, positive, truncate] {
                      float d = nd(gen);
                      if (truncate) {
                        if (std::abs(d) > 100.f) d = 100.f;
                        if (std::abs(d) < 0.001f) d = 0.001f;
                      }
                      return (positive ? std::abs(d) : d);
                    });
    }
  }

  template <DeviceType D, typename T>
  void Transpose2D(const std::string &src_name, const std::string &dst_name) {
    Tensor *input = ws_.GetTensor(src_name);
    Tensor *output = ws_.CreateTensor(dst_name, GetDeviceAllocator(D),
                                      DataTypeToEnum<T>::v());
    const std::vector<index_t> input_shape = input->shape();
    MACE_CHECK(input_shape.size() == 2, "input shape != 2");
    output->Resize({input_shape[1], input_shape[0]});
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();
    for (index_t i = 0; i < input_shape[0]; ++i) {
      for (index_t j = 0; j < input_shape[1]; ++j) {
        output_data[j * input_shape[0] + i] =
            input_data[i * input_shape[1] + j];
      }
    }
  }

  template <DeviceType D, typename T>
  void TransformDataFormat(const std::string &src_name,
                           const DataFormat src_format,
                           const std::string &dst_name,
                           const DataFormat dst_format) {
    Tensor *input = ws_.GetTensor(src_name);
    Tensor *output = ws_.CreateTensor(dst_name, GetDeviceAllocator(D),
                                      DataTypeToEnum<T>::v());
    const std::vector<index_t> input_shape = input->shape();
    MACE_CHECK(input_shape.size() == 4, "input shape != 4");

    if (src_format == NHWC && dst_format == NCHW) {
      index_t batch = input_shape[0];
      index_t height = input_shape[1];
      index_t width = input_shape[2];
      index_t channels = input_shape[3];
      output->Resize({batch, channels, height, width});
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t h = 0; h < height; ++h) {
            for (index_t w = 0; w < width; ++w) {
              output_data[((b * channels + c) * height + h) * width + w] =
                  input_data[((b * height + h) * width + w) * channels + c];
            }
          }
        }
      }
    } else if (src_format == NCHW && dst_format == NHWC) {
      index_t batch = input_shape[0];
      index_t channels = input_shape[1];
      index_t height = input_shape[2];
      index_t width = input_shape[3];
      output->Resize({batch, height, width, channels});
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t b = 0; b < batch; ++b) {
        for (index_t h = 0; h < height; ++h) {
          for (index_t w = 0; w < width; ++w) {
            for (index_t c = 0; c < channels; ++c) {
              output_data[((b * height + h) * width + w) * channels + c] =
                  input_data[((b * channels + c) * height + h) * width + w];
            }
          }
        }
      }
    } else if (src_format == HWOI && dst_format == OIHW) {
      index_t height = input_shape[0];
      index_t width = input_shape[1];
      index_t out_channels = input_shape[2];
      index_t in_channels = input_shape[3];
      index_t hw = height * width;
      index_t oi = out_channels * in_channels;
      output->Resize({out_channels, in_channels, height, width});
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t i = 0; i < oi; ++i) {
        for (index_t j = 0; j < hw; ++j) {
          output_data[i * height * width + j] =
              input_data[j * out_channels * in_channels + i];
        }
      }
    } else if (src_format == OIHW && dst_format == HWOI) {
      index_t out_channels = input_shape[0];
      index_t in_channels = input_shape[1];
      index_t height = input_shape[2];
      index_t width = input_shape[3];
      index_t hw = height * width;
      index_t oi = out_channels * in_channels;
      output->Resize({height, width, out_channels, in_channels});
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t i = 0; i < hw; ++i) {
        for (index_t j = 0; j < oi; ++j) {
          output_data[i * out_channels * in_channels + j] =
              input_data[j * height * width + i];
        }
      }
    } else if (src_format == HWIO && dst_format == OIHW) {
      index_t height = input_shape[0];
      index_t width = input_shape[1];
      index_t in_channels = input_shape[2];
      index_t out_channels = input_shape[3];
      index_t hw = height * width;
      output->Resize({out_channels, in_channels, height, width});
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t m = 0; m < out_channels; ++m) {
        for (index_t c = 0; c < in_channels; ++c) {
          for (index_t k = 0; k < hw; ++k) {
            output_data[((m * in_channels) + c) * height * width + k] =
                input_data[k * out_channels * in_channels + c * out_channels +
                           m];
          }
        }
      }
    } else if (src_format == OHWI && dst_format == OIHW) {
      index_t out_channels = input_shape[0];
      index_t height = input_shape[1];
      index_t width = input_shape[2];
      index_t in_channels = input_shape[3];
      output->Resize({out_channels, in_channels, height, width});
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t b = 0; b < out_channels; ++b) {
        for (index_t c = 0; c < in_channels; ++c) {
          for (index_t h = 0; h < height; ++h) {
            for (index_t w = 0; w < width; ++w) {
              output_data[((b * in_channels + c) * height + h) * width + w] =
                  input_data[((b * height + h) * width + w) * in_channels + c];
            }
          }
        }
      }
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  template <DeviceType D, typename T>
  void FillNHWCInputToNCHWInput(const std::string &name_nchw,
                                const std::string &name_nhwc) {
    Tensor *input = ws_.GetTensor(name_nhwc);
    Tensor *output = ws_.CreateTensor(name_nchw, GetDeviceAllocator(D),
                                      DataTypeToEnum<T>::v());
    const std::vector<index_t> input_shape = input->shape();
    index_t batch = input_shape[0];
    index_t height = input_shape[1];
    index_t width = input_shape[2];
    index_t channels = input_shape[3];
    output->Resize({batch, channels, height, width});
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        for (index_t h = 0; h < height; ++h) {
          for (index_t w = 0; w < width; ++w) {
            output_data[((b * channels + c) * height + h) * width + w] =
                input_data[((b * height + h) * width + w) * channels + c];
          }
        }
      }
    }
  }

  OperatorDef *NewOperatorDef() {
    op_defs_.clear();
    op_defs_.emplace_back(OperatorDef());
    return &op_defs_[op_defs_.size() - 1];
  }

  OperatorDef *AddNewOperatorDef() {
    op_defs_.emplace_back(OperatorDef());
    return &op_defs_[op_defs_.size() - 1];
  }

  Workspace *ws() { return &ws_; }

  bool Setup(DeviceType device) {
    NetDef net_def;
    for (auto &op_def_ : op_defs_) {
      net_def.add_op()->CopyFrom(op_def_);
    }
    net_ = CreateNet(op_registry_, net_def, &ws_, device);
    device_ = device;
    return net_ != nullptr;
  }

  MaceStatus Run() {
    MACE_CHECK_NOTNULL(net_);
    return net_->Run();
  }

  // DEPRECATED(liyin):
  // Test and benchmark should setup model once and run multiple times.
  // Setup time should not be counted during benchmark.
  MaceStatus RunOp(DeviceType device) {
    Setup(device);
    return Run();
  }

  // DEPRECATED(liyin):
  // Test and benchmark should setup model once and run multiple times.
  // Setup time should not be counted during benchmark.
  MaceStatus RunOp() { return RunOp(DeviceType::CPU); }

  MaceStatus RunNet(const NetDef &net_def, const DeviceType device) {
    device_ = device;
    net_ = CreateNet(op_registry_, net_def, &ws_, device, NetMode::INIT);
    MACE_RETURN_IF_ERROR(net_->Run());
    net_ = CreateNet(op_registry_, net_def, &ws_, device);
    return net_->Run();
  }

  Tensor *GetOutput(const char *output_name) {
    return ws_.GetTensor(output_name);
  }

  Tensor *GetTensor(const char *tensor_name) {
    return ws_.GetTensor(tensor_name);
  }

  void Sync() {
    if (net_ && device_ == DeviceType::GPU) {
      OpenCLRuntime::Global()->command_queue().finish();
    }
  }

 public:
  std::shared_ptr<OperatorRegistryBase> op_registry_;
  Workspace ws_;
  std::vector<OperatorDef> op_defs_;
  std::unique_ptr<NetBase> net_;
  DeviceType device_;
};

class OpsTestBase : public ::testing::Test {
 protected:
  virtual void SetUp() {
    // OpenCLRuntime::CreateGlobal();
  }

  virtual void TearDown() {
    // OpenCLRuntime::DestroyGlobal();
  }
};

template <typename T>
void GenerateRandomRealTypeData(const std::vector<index_t> &shape,
                                std::vector<T> *res,
                                bool positive = true) {
  MACE_CHECK_NOTNULL(res);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);

  index_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<index_t>());
  res->resize(size);

  if (DataTypeToEnum<T>::value == DT_HALF) {
    std::generate(res->begin(), res->end(), [&gen, &nd, positive] {
      return half_float::half_cast<half>(positive ? std::abs(nd(gen))
                                                  : nd(gen));
    });
  } else {
    std::generate(res->begin(), res->end(), [&gen, &nd, positive] {
      return positive ? std::abs(nd(gen)) : nd(gen);
    });
  }
}

template <typename T>
void GenerateRandomIntTypeData(const std::vector<index_t> &shape,
                               std::vector<T> *res,
                               const T a = 0,
                               const T b = std::numeric_limits<T>::max()) {
  MACE_CHECK_NOTNULL(res);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> nd(a, b);

  index_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<index_t>());
  res->resize(size);

  std::generate(res->begin(), res->end(), [&gen, &nd] { return nd(gen); });
}

template <typename T>
std::vector<T> VectorStaticCast(const std::vector<float> &&src) {
  std::vector<T> dest;
  dest.reserve(src.size());
  for (float f : src) {
    dest.push_back(static_cast<T>(f));
  }
  return std::move(dest);
}

template <typename T>
std::unique_ptr<Tensor> CreateTensor(const std::vector<index_t> &shape,
                                     const std::vector<T> &data) {
  std::unique_ptr<Tensor> res(
      new Tensor(GetDeviceAllocator(DeviceType::CPU), DataTypeToEnum<T>::v()));
  res->Resize(shape);
  T *input_data = res->mutable_data<T>();
  memcpy(input_data, data.data(), data.size() * sizeof(T));
  return res;
}

inline bool IsSameSize(const Tensor &x, const Tensor &y) {
  if (x.dim_size() != y.dim_size()) return false;
  for (int d = 0; d < x.dim_size(); ++d) {
    if (x.dim(d) != y.dim(d)) return false;
  }
  return true;
}

inline std::string ShapeToString(const Tensor &x) {
  std::stringstream stream;
  for (int i = 0; i < x.dim_size(); i++) {
    if (i > 0) stream << ",";
    int64_t dim = x.dim(i);
    if (dim < 0) {
      stream << "?";
    } else {
      stream << dim;
    }
  }
  stream << "]";
  return std::string(stream.str());
}

template <typename T>
struct is_floating_point_type {
  static const bool value = std::is_same<T, float>::value ||
                            std::is_same<T, double>::value ||
                            std::is_same<T, half>::value;
};

template <typename T>
inline void ExpectEqual(const T &a, const T &b) {
  EXPECT_EQ(a, b);
}

template <>
inline void ExpectEqual<float>(const float &a, const float &b) {
  EXPECT_FLOAT_EQ(a, b);
}

template <>
inline void ExpectEqual<double>(const double &a, const double &b) {
  EXPECT_DOUBLE_EQ(a, b);
}

inline void AssertSameDims(const Tensor &x, const Tensor &y) {
  ASSERT_TRUE(IsSameSize(x, y)) << "x.shape [" << ShapeToString(x) << "] vs "
                                << "y.shape [ " << ShapeToString(y) << "]";
}

template <typename EXP_TYPE,
          typename RES_TYPE,
          bool is_fp = is_floating_point_type<EXP_TYPE>::value>
struct Expector;

// Partial specialization for float and double.
template <typename EXP_TYPE, typename RES_TYPE>
struct Expector<EXP_TYPE, RES_TYPE, true> {
  static void Equal(const EXP_TYPE &a, const RES_TYPE &b) { ExpectEqual(a, b); }

  static void Equal(const Tensor &x, const Tensor &y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<EXP_TYPE>::v());
    ASSERT_EQ(y.dtype(), DataTypeToEnum<RES_TYPE>::v());
    AssertSameDims(x, y);
    Tensor::MappingGuard x_mapper(&x);
    Tensor::MappingGuard y_mapper(&y);
    auto a = x.data<EXP_TYPE>();
    auto b = y.data<RES_TYPE>();
    for (int i = 0; i < x.size(); ++i) {
      ExpectEqual(a[i], b[i]);
    }
  }

  static void Near(const Tensor &x,
                   const Tensor &y,
                   const double rel_err,
                   const double abs_err) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<EXP_TYPE>::v());
    ASSERT_EQ(y.dtype(), DataTypeToEnum<RES_TYPE>::v());
    AssertSameDims(x, y);
    Tensor::MappingGuard x_mapper(&x);
    Tensor::MappingGuard y_mapper(&y);
    auto a = x.data<EXP_TYPE>();
    auto b = y.data<RES_TYPE>();
    if (x.dim_size() == 4) {
      for (int n = 0; n < x.dim(0); ++n) {
        for (int h = 0; h < x.dim(1); ++h) {
          for (int w = 0; w < x.dim(2); ++w) {
            for (int c = 0; c < x.dim(3); ++c) {
              const double error = abs_err + rel_err * std::abs(*a);
              EXPECT_NEAR(*a, *b, error) << "with index = [" << n << ", " << h
                                         << ", " << w << ", " << c << "]";
              a++;
              b++;
            }
          }
        }
      }
    } else {
      for (int i = 0; i < x.size(); ++i) {
        const double error = abs_err + rel_err * std::abs(a[i]);
        EXPECT_NEAR(a[i], b[i], error) << "a = " << a << " b = " << b
                                       << " index = " << i;
      }
    }
  }
};

template <typename EXP_TYPE, typename RES_TYPE>
struct Expector<EXP_TYPE, RES_TYPE, false> {
  static void Equal(const EXP_TYPE &a, const RES_TYPE &b) { ExpectEqual(a, b); }

  static void Equal(const Tensor &x, const Tensor &y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<EXP_TYPE>::v());
    ASSERT_EQ(y.dtype(), DataTypeToEnum<RES_TYPE>::v());
    AssertSameDims(x, y);
    Tensor::MappingGuard x_mapper(&x);
    Tensor::MappingGuard y_mapper(&y);
    auto a = x.data<EXP_TYPE>();
    auto b = y.data<RES_TYPE>();
    for (int i = 0; i < x.size(); ++i) {
      ExpectEqual(a[i], b[i]);
    }
  }

  static void Near(const Tensor &x,
                   const Tensor &y,
                   const double rel_err,
                   const double abs_err) {
    MACE_UNUSED(rel_err);
    MACE_UNUSED(abs_err);
    Equal(x, y);
  }
};

template <typename T>
void ExpectTensorNear(const Tensor &x,
                      const Tensor &y,
                      const double rel_err = 1e-5,
                      const double abs_err = 1e-8) {
  Expector<T, T>::Near(x, y, rel_err, abs_err);
}

template <typename EXP_TYPE, typename RES_TYPE>
void ExpectTensorNear(const Tensor &x,
                      const Tensor &y,
                      const double rel_err = 1e-5,
                      const double abs_err = 1e-8) {
  Expector<EXP_TYPE, RES_TYPE>::Near(x, y, rel_err, abs_err);
}

template <typename T>
void ExpectTensorSimilar(const Tensor &x,
                         const Tensor &y,
                         const double abs_err = 1e-5) {
  AssertSameDims(x, y);
  Tensor::MappingGuard x_mapper(&x);
  Tensor::MappingGuard y_mapper(&y);
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  double dot_product = 0.0, x_norm = 0.0, y_norm = 0.0;
  for (index_t i = 0; i < x.size(); i++) {
    dot_product += x_data[i] * y_data[i];
    x_norm += x_data[i] * x_data[i];
    y_norm += y_data[i] * y_data[i];
  }
  double similarity = dot_product / (sqrt(x_norm) * sqrt(y_norm));
  EXPECT_NEAR(1.0, similarity, abs_err);
}

template <DeviceType D, typename T>
void BufferToImage(OpsTestNet *net,
                   const std::string &input_name,
                   const std::string &output_name,
                   const kernels::BufferType type,
                   const int wino_block_size = 2) {
  MACE_CHECK_NOTNULL(net);

  OpDefBuilder("BufferToImage", "BufferToImageTest")
    .Input(input_name)
    .Output(output_name)
    .AddIntArg("buffer_type", type)
    .AddIntArg("wino_block_size", wino_block_size)
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
    .Finalize(net->NewOperatorDef());

  // Run
  net->RunOp(D);

  net->Sync();
}

template <DeviceType D, typename T>
void ImageToBuffer(OpsTestNet *net,
                   const std::string &input_name,
                   const std::string &output_name,
                   const kernels::BufferType type,
                   const int wino_block_size = 2) {
  MACE_CHECK_NOTNULL(net);

  OpDefBuilder("ImageToBuffer", "ImageToBufferTest")
    .Input(input_name)
    .Output(output_name)
    .AddIntArg("buffer_type", type)
    .AddIntArg("wino_block_size", wino_block_size)
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
    .Finalize(net->NewOperatorDef());

  // Run
  net->RunOp(D);

  net->Sync();
}

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPS_TEST_UTIL_H_
