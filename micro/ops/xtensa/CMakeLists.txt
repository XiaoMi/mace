add_library(micro_ops_xtensa
  conv_2d_xtensa.cc
  depthwise_conv_2d_xtensa.cc
  matmul_xtensa.cc
)

target_link_libraries(micro_ops_xtensa
  PRIVATE micro_base
)

target_include_directories(micro_ops_xtensa PRIVATE ${XA_NNLIB_INCLUDE_DIR})

target_link_libraries(micro_all_ops INTERFACE micro_ops_xtensa)

set(combine_xtensa_stamp ${CMAKE_CURRENT_BINARY_DIR}/combine_xtensa.stamp)
add_custom_command(
  OUTPUT ${combine_xtensa_stamp}
  COMMAND touch ${combine_xtensa_stamp}
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_ops_xtensa>\""
  DEPENDS micro_ops_xtensa
  COMMENT "Combining xtensa libs ..."
)

add_custom_target(micro_combine_xtensa
  ALL DEPENDS ${combine_xtensa_stamp}
)
