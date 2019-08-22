set(CAFFE_PROTO_PROTOS ${PROJECT_SOURCE_DIR}/third_party/caffe/caffe.proto)
set(MACE_PROTO_PYTHON_DIR ${PROJECT_SOURCE_DIR}/tools/python/py_proto)

foreach(proto_file ${CAFFE_PROTO_PROTOS})
    get_filename_component(proto_file_abs ${proto_file} ABSOLUTE)
    get_filename_component(basename ${proto_file} NAME_WE)
    set(PROTO_GENERATED_PY_FILES ${MACE_PROTO_PYTHON_DIR}/${basename}_pb2.py)

    add_custom_command(
            OUTPUT ${PROTO_GENERATED_PY_FILES}
            COMMAND ${PROTOC_BIN} --python_out ${MACE_PROTO_PYTHON_DIR} -I ${PROJECT_SOURCE_DIR}/third_party/caffe ${proto_file_abs}
            COMMENT "Generating ${PROTO_GENERATED_PY_FILES} from ${proto_file}"
            DEPENDS protoc_bin
            VERBATIM
    )
endforeach()

add_custom_target(caffe_proto_src ALL DEPENDS ${PROTO_GENERATED_PY_FILES})
