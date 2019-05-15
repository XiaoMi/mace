set(NNLIB_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/nnlib")
set(NNLIB_INCLUDE_DIR  "${NNLIB_INSTALL_DIR}")

include_directories(SYSTEM "${NNLIB_INCLUDE_DIR}")

set(NNLIB_CONTROLLER "${NNLIB_INSTALL_DIR}/${ANDROID_ABI}/libhexagon_controller.so")
add_library(hexagon_controller SHARED IMPORTED GLOBAL)
set_target_properties(hexagon_controller PROPERTIES IMPORTED_LOCATION ${NNLIB_CONTROLLER})

install(FILES ${NNLIB_CONTROLLER} DESTINATION lib)
