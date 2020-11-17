function(compilerSpecificCompileOptions PROJECTNAME ROOT)
target_compile_options(${PROJECTNAME}
  PRIVATE "-Wno-unused-parameter"
  PRIVATE "-Wno-sign-compare"
  PRIVATE "-Wno-strict-aliasing"
  PRIVATE "-Wno-unused-variable"
)
endfunction()
set(ROOT ${MACE_MICRO_THIRD_PARTY_PATH}/CMSIS_5)

include_directories(${ROOT}/CMSIS/Core/Include)

add_subdirectory(${ROOT}/CMSIS/DSP/Source "CMSIS_DSP" EXCLUDE_FROM_ALL)
add_subdirectory(${ROOT}/CMSIS/NN/Source "CMSIS_NN"  EXCLUDE_FROM_ALL)

target_include_directories(CMSISDSP INTERFACE ${ROOT}/CMSIS/Core/Include)
target_include_directories(CMSISNN INTERFACE ${ROOT}/CMSIS/Core/Include)
include_directories(third_party/CMSIS_5/CMSIS/Core/Include)

add_library(micro_ops_cmsis_nn
  arm_conv_2d_int8.cc
  arm_pooling_int8.cc
  arm_softmax_int8.cc
  arm_mat_mul_int8.cc
  arm_eltwise_int8.cc
  arm_depthwise_conv_2d_int8.cc
  dequantize.cc
  quantize.cc
  utilities.cc
)

target_link_libraries(micro_ops_cmsis_nn
  PRIVATE micro_ops # Depends on ops base
  PRIVATE micro_base
  PRIVATE CMSISNN
)

target_link_libraries(micro_all_ops INTERFACE micro_ops_cmsis_nn)

set(combine_cmsis_stamp ${CMAKE_CURRENT_BINARY_DIR}/combine_cmsis.stamp)
add_custom_command(
  OUTPUT ${combine_cmsis_stamp}
  COMMAND touch ${combine_cmsis_stamp}
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_ops_cmsis_nn>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNReshape>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNBasicMaths>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNConcatenation>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNFullyConnected>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNConvolutions>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNActivation>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNPooling>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNSoftmax>\""
  COMMAND ${MACE_MICRO_TOOLS_PATH}/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:CMSISNNSupport>\""
  DEPENDS micro_ops_cmsis_nn CMSISNN
  COMMENT "Combining cmsis libs ..."
)

add_custom_target(micro_combine_cmsis
  ALL DEPENDS ${combine_cmsis_stamp}
)

add_dependencies(micro micro_combine_cmsis)