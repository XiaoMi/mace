set(MACE_PROTO_PROTOS mace.proto)
set(MACE_PROTO_SRCS)
set(MACE_PROTO_HDRS)

foreach(proto_file ${MACE_PROTO_PROTOS})
  get_filename_component(proto_file_abs ${proto_file} ABSOLUTE)
  get_filename_component(basename ${proto_file} NAME_WE)
  set(PROTO_GENERATED_FILES ${basename}.pb.cc ${basename}.pb.h)

  list(APPEND MACE_PROTO_SRCS ${basename}.pb.cc)
  list(APPEND MACE_PROTO_HDRS ${basename}.pb.h)

  add_custom_command(
      OUTPUT ${PROTO_GENERATED_FILES}
      COMMAND ${PROTOC_BIN} --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} ${proto_file_abs}
      COMMENT "Generating ${PROTO_GENERATED_FILES} from ${proto_file}"
      DEPENDS protoc_bin
      VERBATIM
  )
endforeach()

add_custom_target(mace_proto_src DEPENDS ${MACE_PROTO_SRCS}
  COMMENT "Checking if re-generation is required" )

add_library(proto ${MACE_PROTO_SRCS})
target_link_libraries(proto libprotobuf_lite)

install(TARGETS proto ARCHIVE DESTINATION lib)
