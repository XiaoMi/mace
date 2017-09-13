//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_TEST_UTIL_H_
#define MACE_OPS_TEST_UTIL_H_

#include <type_traits>

#include "gtest/gtest.h"
#include "mace/core/common.h"
#include "mace/core/net.h"
#include "mace/core/tensor.h"

namespace mace {

class OpDefBuilder {
  public:
    OpDefBuilder(const char* type, const char* name) {
      op_def_.set_type(type);
      op_def_.set_name(name);
    }
    OpDefBuilder& Input(const char* input_name) {
      op_def_.add_input(input_name);
      return *this;
    }
    OpDefBuilder& Output(const char* output_name) {
      op_def_.add_output(output_name);
      return *this;
    }
    void Finalize(OperatorDef* op_def) const {
      MACE_CHECK(op_def != nullptr, "input should not be null.");
      *op_def = op_def_;
    }
    OperatorDef op_def_;
};

class OpsTestNet {
  public:
    template <typename T>
    void AddInputFromArray(const char* name,
                           const std::vector<index_t>& shape,
                           const std::vector<T>& data) {
      Tensor* input = ws_.CreateTensor(name, cpu_allocator(), DataTypeToEnum<T>::v());
      input->Resize(shape);
      float* input_data = input->mutable_data<float>();
      // TODO check the dims
      memcpy(input_data, data.data(), data.size() * sizeof(T));
    }

    void AddIntArg(const char* name, const int value) {
      auto arg = op_def_.add_arg();
      arg->set_name(name);
      arg->set_i(value);
    }

    void AddFloatArg(const char* name, const float value) {
      auto arg = op_def_.add_arg();
      arg->set_name(name);
      arg->set_f(value);
    }

    void AddStringArg(const char* name, const char* value) {
      auto arg = op_def_.add_arg();
      arg->set_name(name);
      arg->set_s(value);
    }

    void AddIntsArg(const char* name, const std::vector<int>& values) {
      auto arg = op_def_.add_arg();
      arg->set_name(name);
      for (auto value : values) {
        arg->add_ints(value);
      }
    }

    void AddFloatsArg(const char* name, const std::vector<float>& values) {
      auto arg = op_def_.add_arg();
      arg->set_name(name);
      for (auto value : values) {
        arg->add_floats(value);
      }
    }

    void AddStringsArg(const char* name, const std::vector<const char*>& values) {
      auto arg = op_def_.add_arg();
      arg->set_name(name);
      for (auto value : values) {
        arg->add_strings(value);
      }
    }

    OperatorDef* operator_def() { return &op_def_; }

    Workspace* ws() { return &ws_; }

    bool RunOp(DeviceType device) {
      if (!net_) {
        NetDef net_def;
        net_def.add_op()->CopyFrom(op_def_);
        VLOG(0) << net_def.DebugString();
        net_ = CreateNet(net_def, &ws_, device);
      }
      return net_->Run();
    }

    bool RunOp() {
      return RunOp(DeviceType::CPU);
    }

    Tensor* GetOutput(const char* output_name) {
      return ws_.GetTensor(output_name);
    }

  public:
    Workspace ws_;
    OperatorDef op_def_;
    std::unique_ptr<NetBase> net_;
};

class OpsTestBase : public ::testing::Test {
  public:
    OpsTestNet* test_net() { return &test_net_; };

  protected:
    virtual void TearDown() {
      auto ws = test_net_.ws();
      auto tensor_names = ws->Tensors();
      for (auto& name : tensor_names) {
        ws->RemoveTensor(name);
      }
    }

  private:
    OpsTestNet test_net_;
};

template <typename T>
Tensor CreateTensor(const std::vector<index_t>& shape, const std::vector<T>& data) {
  Tensor res(cpu_allocator(), DataTypeToEnum<T>::v());
  res.Resize(shape);
  float* input_data = res.mutable_data<float>();
  memcpy(input_data, data.data(), data.size() * sizeof(T));
  return res;
}

inline bool IsSameSize(const Tensor& x, const Tensor& y) {
  if (x.dim_size() != y.dim_size()) return false;
  for (int d = 0; d < x.dim_size(); ++d) {
    if (x.dim(d) != y.dim(d)) return false;
  }
  return true;
}

inline std::string ShapeToString(const Tensor& x) {
  std::stringstream stream;
  for (int i = 0; i < x.dim_size(); i++) {
    if (i > 0) stream<<",";
    int64_t dim = x.dim(i);
    if (dim < 0) {
      stream<<"?";
    } else {
      stream<<dim;
    }
  }
  stream<<"]";
  return std::string(stream.str());
}


template <typename T>
struct is_floating_point_type {
  static const bool value = std::is_same<T, float>::value ||
                            std::is_same<T, double>::value;
};

template <typename T>
inline void ExpectEqual(const T& a, const T& b) {
  EXPECT_EQ(a, b);
}

template <>
inline void ExpectEqual<float>(const float& a, const float& b) {
  EXPECT_FLOAT_EQ(a, b);
}

template <>
inline void ExpectEqual<double>(const double& a, const double& b) {
  EXPECT_DOUBLE_EQ(a, b);
}

inline void AssertSameTypeDims(const Tensor& x, const Tensor& y) {
  ASSERT_EQ(x.dtype(), y.dtype());
  ASSERT_TRUE(IsSameSize(x, y))
        << "x.shape [" << ShapeToString(x) << "] vs "
        << "y.shape [ " << ShapeToString(y) << "]";
}

template <typename T, bool is_fp = is_floating_point_type<T>::value>
struct Expector;
// Partial specialization for float and double.
template <typename T>
struct Expector<T, true> {
  static void Equal(const T& a, const T& b) { ExpectEqual(a, b); }

  static void Equal(const Tensor& x, const Tensor& y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<T>::v());
    AssertSameTypeDims(x, y);
    auto a = x.data<T>();
    auto b = y.data<T>();
    for (int i = 0; i < x.size(); ++i) {
      ExpectEqual(a(i), b(i));
    }
  }

  static void Near(const Tensor& x, const Tensor& y, const double abs_err) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<T>::v());
    AssertSameTypeDims(x, y);
    auto a = x.data<T>();
    auto b = y.data<T>();
    for (int i = 0; i < x.size(); ++i) {
      EXPECT_NEAR(a[i], b[i], abs_err)
            << "a = " << a << " b = " << b << " index = " << i;
    }
  }
};

template <typename T>
void ExpectTensorNear(const Tensor& x, const Tensor& y, const double abs_err) {
  static_assert(is_floating_point_type<T>::value, "T is not a floating point type");
  Expector<T>::Near(x, y ,abs_err);
}

} // namespace mace

#endif //  MACE_OPS_TEST_UTIL_H_
