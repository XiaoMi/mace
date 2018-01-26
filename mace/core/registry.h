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

  void Register(const SrcType &key, Creator creator) {
    VLOG(2) << "Registering: " << key;
    std::lock_guard<std::mutex> lock(register_mutex_);
    MACE_CHECK(registry_.count(key) == 0, "Key already registered: ", key);
    registry_[key] = creator;
  }

  inline bool Has(const SrcType &key) const {
    return registry_.count(key) != 0;
  }

  unique_ptr<ObjectType> Create(const SrcType &key, Args... args) const {
    if (registry_.count(key) == 0) {
      LOG(FATAL) << "Key not registered: " << key;
    }
    return registry_.at(key)(args...);
  }

  /**
   * Returns the keys currently registered as a vector.
   */
  vector<SrcType> Keys() const {
    vector<SrcType> keys;
    for (const auto &it : registry_) {
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
  Registerer(const SrcType &key,
             Registry<SrcType, ObjectType, Args...> *registry,
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
  Registry<SrcType, ObjectType, ##__VA_ARGS__> *RegistryName();             \
  typedef Registerer<SrcType, ObjectType, ##__VA_ARGS__>                    \
      Registerer##RegistryName;

/*
#define MACE_DEFINE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__> *RegistryName() {           \
    static Registry<SrcType, ObjectType, ##__VA_ARGS__> *registry =        \
        new Registry<SrcType, ObjectType, ##__VA_ARGS__>();                \
    return registry;                                                       \
  }
*/

#define MACE_DECLARE_REGISTRY(RegistryName, ObjectType, ...)         \
  MACE_DECLARE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                              ##__VA_ARGS__)

/*
#define MACE_DEFINE_REGISTRY(RegistryName, ObjectType, ...)         \
  MACE_DEFINE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                             ##__VA_ARGS__)
*/

#define MACE_REGISTER_TYPED_CLASS(RegistryName, registry, key, ...)   \
  Registerer##RegistryName MACE_ANONYMOUS_VARIABLE(l_##RegistryName)( \
      key, registry, Registerer##RegistryName::DefaultCreator<__VA_ARGS__>);

#define MACE_REGISTER_CLASS(RegistryName, registry, key, ...) \
  MACE_REGISTER_TYPED_CLASS(RegistryName, registry, key, __VA_ARGS__)

}  // namespace mace

#endif  // MACE_CORE_REGISTRY_H_
