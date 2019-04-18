set(HTA_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/hta")
set(HTA_INCLUDE_DIR  "${HTA_INSTALL_DIR}")

include_directories(SYSTEM "${HTA_INCLUDE_DIR}")

set(HTA_CONTROLLER
  "${HTA_INSTALL_DIR}/${ANDROID_ABI}/libhta_controller.so"
)
set(HTA_RUNTIME
  "${HTA_INSTALL_DIR}/${ANDROID_ABI}/libhta_hexagon_runtime.so"
)
set(HTA_NPU
  "${HTA_INSTALL_DIR}/${ANDROID_ABI}/libnpu.so"
)

add_library(hta_controller SHARED IMPORTED GLOBAL)
add_library(hta_hexagon_runtime SHARED IMPORTED GLOBAL)
add_library(npu SHARED IMPORTED GLOBAL)
set_target_properties(hta_controller PROPERTIES IMPORTED_LOCATION ${HTA_CONTROLLER})
set_target_properties(hta_hexagon_runtime PROPERTIES IMPORTED_LOCATION ${HTA_RUNTIME})
set_target_properties(npu PROPERTIES IMPORTED_LOCATION ${HTA_NPU})

install(FILES ${HTA_CONTROLLER} DESTINATION lib)
install(FILES ${HTA_RUNTIME} DESTINATION lib)
install(FILES ${HTA_NPU} DESTINATION lib)
