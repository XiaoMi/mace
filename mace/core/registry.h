//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_REGISTRY_H_
#define MACE_CORE_REGISTRY_H_

#include <mutex>

namespace mace {

template <class SrcType, class ObjectType, class... Args>
class Registry {
 public:
  typedef std::function<std::unique_ptr<ObjectType>(Args...)> Creator;

  Registry() : registry_() {}

  void Register(const SrcType& key, Creator creator) {
    std::lock_guard<std::mutex> lock(register_mutex_);
    MACE_CHECK(registry_.count(key) == 0, "Key already registered.");
    registry_[key] = creator;
  }

  inline bool Has(const SrcType& key) { return registry_.count(key) != 0; }

  unique_ptr<ObjectType> Create(const SrcType& key, Args... args) {
    if (registry_.count(key) == 0) {
      VLOG(2) << "Key not registered: " << key;
      return nullptr;
    }
    return registry_[key](args...);
  }

  /**
   * Returns the keys currently registered as a vector.
   */
  vector<SrcType> Keys() {
    vector<SrcType> keys;
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

 private:
  std::map<SrcType, Creator> registry_;
  std::mutex register_mutex_;

  DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class SrcType, class ObjectType, class... Args>
class Registerer {
 public:
  Registerer(const SrcType& key,
             Registry<SrcType, ObjectType, Args...>* registry,
             typename Registry<SrcType, ObjectType, Args...>::Creator creator) {
    registry->Register(key, creator);
  }

  template <class DerivedType>
  static unique_ptr<ObjectType> DefaultCreator(Args... args) {
    return std::unique_ptr<ObjectType>(new DerivedType(args...));
  }
};

#define MACE_CONCATENATE_IMPL(s1, s2) s1##s2
#define MACE_CONCATENATE(s1, s2) MACE_CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
#define MACE_ANONYMOUS_VARIABLE(str) MACE_CONCATENATE(str, __COUNTER__)
#else
#define MACE_ANONYMOUS_VARIABLE(str) MACE_CONCATENATE(str, __LINE__)
#endif

#define MACE_DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName();             \
  typedef Registerer<SrcType, ObjectType, ##__VA_ARGS__>                    \
      Registerer##RegistryName;

#define MACE_DEFINE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName() {           \
    static Registry<SrcType, ObjectType, ##__VA_ARGS__>* registry =        \
        new Registry<SrcType, ObjectType, ##__VA_ARGS__>();                \
    return registry;                                                       \
  }

#define MACE_DECLARE_REGISTRY(RegistryName, ObjectType, ...)         \
  MACE_DECLARE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                              ##__VA_ARGS__)

#define MACE_DEFINE_REGISTRY(RegistryName, ObjectType, ...)         \
  MACE_DEFINE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                             ##__VA_ARGS__)

#define MACE_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  namespace {                                                                \
  static Registerer##RegistryName MACE_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), __VA_ARGS__);

#define MACE_REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  namespace {                                                                \
  static Registerer##RegistryName MACE_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(),                                                   \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>);                \
  }

#define MACE_REGISTER_CREATOR(RegistryName, key, ...) \
  MACE_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

#define MACE_REGISTER_CLASS(RegistryName, key, ...) \
  MACE_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

}  // namespace mace

#endif  // MACE_CORE_REGISTRY_H_
