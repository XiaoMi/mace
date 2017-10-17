//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_PROTO_UTILS_H_
#define MACE_CORE_PROTO_UTILS_H_

#include <map>

#include "google/protobuf/message_lite.h"
#ifndef MACE_USE_LITE_PROTO
#include "google/protobuf/message.h"
#endif  // !MACE_USE_LITE_PROTO

#include "mace/core/common.h"
#include "mace/proto/mace.pb.h"

namespace mace {

using std::string;
using ::google::protobuf::MessageLite;

// Common interfaces that reads file contents into a string.
bool ReadStringFromFile(const char *filename, string *str);
bool WriteStringToFile(const string &str, const char *filename);

// Common interfaces that are supported by both lite and full protobuf.
bool ReadProtoFromBinaryFile(const char *filename, MessageLite *proto);
inline bool ReadProtoFromBinaryFile(const string filename, MessageLite *proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const MessageLite &proto, const char *filename);
inline void WriteProtoToBinaryFile(const MessageLite &proto,
                                   const string &filename) {
  return WriteProtoToBinaryFile(proto, filename.c_str());
}

#ifdef MACE_USE_LITE_PROTO

inline string ProtoDebugString(const MessageLite &proto) {
  return proto.SerializeAsString();
}

// Text format MessageLite wrappers: these functions do nothing but just
// allowing things to compile. It will produce a runtime error if you are using
// MessageLite but still want text support.
inline bool ReadProtoFromTextFile(const char * /*filename*/,
                                  MessageLite * /*proto*/) {
  LOG(FATAL) << "If you are running lite version, you should not be "
             << "calling any text-format protobuffers.";
  return false;  // Just to suppress compiler warning.
}
inline bool ReadProtoFromTextFile(const string filename, MessageLite *proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void WriteProtoToTextFile(const MessageLite & /*proto*/,
                                 const char * /*filename*/) {
  LOG(FATAL) << "If you are running lite version, you should not be "
             << "calling any text-format protobuffers.";
}
inline void WriteProtoToTextFile(const MessageLite &proto,
                                 const string &filename) {
  return WriteProtoToTextFile(proto, filename.c_str());
}

inline bool ReadProtoFromFile(const char *filename, MessageLite *proto) {
  return (ReadProtoFromBinaryFile(filename, proto) ||
          ReadProtoFromTextFile(filename, proto));
}

inline bool ReadProtoFromFile(const string &filename, MessageLite *proto) {
  return ReadProtoFromFile(filename.c_str(), proto);
}

#else  // MACE_USE_LITE_PROTO

using ::google::protobuf::Message;

inline string ProtoDebugString(const Message &proto) {
  return proto.ShortDebugString();
}

bool ReadProtoFromTextFile(const char *filename, Message *proto);
inline bool ReadProtoFromTextFile(const string filename, Message *proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message &proto, const char *filename);
inline void WriteProtoToTextFile(const Message &proto, const string &filename) {
  return WriteProtoToTextFile(proto, filename.c_str());
}

// Read Proto from a file, letting the code figure out if it is text or binary.
inline bool ReadProtoFromFile(const char *filename, Message *proto) {
  return (ReadProtoFromBinaryFile(filename, proto) ||
          ReadProtoFromTextFile(filename, proto));
}

inline bool ReadProtoFromFile(const string &filename, Message *proto) {
  return ReadProtoFromFile(filename.c_str(), proto);
}

#endif  // MACE_USE_LITE_PROTO

template <class IterableInputs = std::initializer_list<string>,
          class IterableOutputs = std::initializer_list<string>,
          class IterableArgs = std::initializer_list<Argument>>
OperatorDef CreateOperatorDef(const string &type,
                              const string &name,
                              const IterableInputs &inputs,
                              const IterableOutputs &outputs,
                              const IterableArgs &args) {
  OperatorDef def;
  def.set_type(type);
  def.set_name(name);
  for (const string &in : inputs) {
    def.add_input(in);
  }
  for (const string &out : outputs) {
    def.add_output(out);
  }
  for (const Argument &arg : args) {
    def.add_arg()->CopyFrom(arg);
  }
  return def;
}

// A simplified version compared to the full CreateOperator, if you do not need
// to specify args.
template <class IterableInputs = std::initializer_list<string>,
          class IterableOutputs = std::initializer_list<string>>
inline OperatorDef CreateOperatorDef(const string &type,
                                     const string &name,
                                     const IterableInputs &inputs,
                                     const IterableOutputs &outputs) {
  return CreateOperatorDef(type, name, inputs, outputs,
                           std::vector<Argument>());
}

/**
 * @brief A helper class to index into arguments.
 *
 * This helper helps us to more easily index into a set of arguments
 * that are present in the operator. To save memory, the argument helper
 * does not copy the operator def, so one would need to make sure that the
 * lifetime of the OperatorDef object outlives that of the ArgumentHelper.
 */
class ArgumentHelper {
 public:
  template <typename Def>
  static bool HasArgument(const Def &def, const string &name) {
    return ArgumentHelper(def).HasArgument(name);
  }

  template <typename Def, typename T>
  static T GetSingleArgument(const Def &def,
                             const string &name,
                             const T &default_value) {
    return ArgumentHelper(def).GetSingleArgument<T>(name, default_value);
  }

  template <typename Def, typename T>
  static bool HasSingleArgumentOfType(const Def &def, const string &name) {
    return ArgumentHelper(def).HasSingleArgumentOfType<T>(name);
  }

  template <typename Def, typename T>
  static vector<T> GetRepeatedArgument(
      const Def &def,
      const string &name,
      const std::vector<T> &default_value = std::vector<T>()) {
    return ArgumentHelper(def).GetRepeatedArgument<T>(name, default_value);
  }

  template <typename Def, typename MessageType>
  static MessageType GetMessageArgument(const Def &def, const string &name) {
    return ArgumentHelper(def).GetMessageArgument<MessageType>(name);
  }

  template <typename Def, typename MessageType>
  static vector<MessageType> GetRepeatedMessageArgument(const Def &def,
                                                        const string &name) {
    return ArgumentHelper(def).GetRepeatedMessageArgument<MessageType>(name);
  }

  explicit ArgumentHelper(const OperatorDef &def);
  explicit ArgumentHelper(const NetDef &netdef);
  bool HasArgument(const string &name) const;

  template <typename T>
  T GetSingleArgument(const string &name, const T &default_value) const;
  template <typename T>
  bool HasSingleArgumentOfType(const string &name) const;
  template <typename T>
  vector<T> GetRepeatedArgument(
      const string &name,
      const std::vector<T> &default_value = std::vector<T>()) const;

  template <typename MessageType>
  MessageType GetMessageArgument(const string &name) const {
    MACE_CHECK(arg_map_.count(name), "Cannot find parameter named " + name);
    MessageType message;
    if (arg_map_.at(name).has_s()) {
      MACE_CHECK(message.ParseFromString(arg_map_.at(name).s()),
                 "Faild to parse content from the string");
    } else {
      VLOG(1) << "Return empty message for parameter " << name;
    }
    return message;
  }

  template <typename MessageType>
  vector<MessageType> GetRepeatedMessageArgument(const string &name) const {
    MACE_CHECK(arg_map_.count(name), "Cannot find parameter named " + name);
    vector<MessageType> messages(arg_map_.at(name).strings_size());
    for (int i = 0; i < messages.size(); ++i) {
      MACE_CHECK(messages[i].ParseFromString(arg_map_.at(name).strings(i)),
                 "Faild to parse content from the string");
    }
    return messages;
  }

 private:
  std::map<string, Argument> arg_map_;
};

const Argument &GetArgument(const OperatorDef &def, const string &name);
bool GetFlagArgument(const OperatorDef &def,
                     const string &name,
                     bool def_value = false);

Argument *GetMutableArgument(const string &name,
                             const bool create_if_missing,
                             OperatorDef *def);

template <typename T>
Argument MakeArgument(const string &name, const T &value);

template <typename T>
inline void AddArgument(const string &name, const T &value, OperatorDef *def) {
  GetMutableArgument(name, true, def)->CopyFrom(MakeArgument(name, value));
}

}  // namespace mace

#endif  // MACE_CORE_PROTO_UTILS_H_
