macro(generate_proto proto_file)
  get_filename_component(proto_file_abs ${proto_file} ABSOLUTE)
  get_filename_component(basename ${proto_file} NAME_WE)

  set(${basename}_proto_files ${basename}.pb.cc ${basename}.pb.h)
  set(${basename}_proto_srcs  ${basename}.pb.cc)

  add_custom_command(
      OUTPUT ${basename}_proto_files
      COMMAND ${PROTOC_BIN} --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} ${proto_file_abs}
      COMMENT "Generating ${basename}_proto_files from ${proto_file}"
      DEPENDS protoc_bin
      VERBATIM
  )

  set(PROTO_PYTHON_DIR ${PROJECT_SOURCE_DIR}/tools/python/py_proto)
  set(PROTO_GENERATED_PY_FILES ${PROTO_PYTHON_DIR}/${basename}_pb2.py)
  add_custom_command(
      OUTPUT ${PROTO_GENERATED_PY_FILES}
      COMMAND ${PROTOC_BIN} --python_out ${PROTO_PYTHON_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} ${proto_file_abs}
      COMMENT "Generating ${PROTO_GENERATED_PY_FILES} from ${proto_file}"
      DEPENDS protoc_bin
      VERBATIM
  )

  add_custom_target(${basename}_proto_cpp DEPENDS ${basename}_proto_files
        COMMENT "Checking if re-generation is required")
  add_custom_target(${basename}_proto_py ALL DEPENDS ${PROTO_GENERATED_PY_FILES})
endmacro()

generate_proto(mace.proto)
generate_proto(micro_mem.proto)

add_library(proto ${mace_proto_srcs})
add_dependencies(proto mace_proto_cpp)
set_source_files_properties(
    ${mace_proto_srcs}
    PROPERTIES GENERATED TRUE
)
target_link_libraries(proto libprotobuf_lite)

install(TARGETS proto ARCHIVE DESTINATION lib)
