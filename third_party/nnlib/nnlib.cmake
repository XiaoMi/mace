set(NNLIB_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/nnlib")
set(NNLIB_INCLUDE_DIR  "${NNLIB_INSTALL_DIR}")

include_directories(SYSTEM "${NNLIB_INCLUDE_DIR}")

if(ANDROID)
set(NNLIB_CONTROLLER "${NNLIB_INSTALL_DIR}/${ANDROID_ABI}/libhexagon_controller.so")
elseif(QNX)
set(NNLIB_CONTROLLER "${NNLIB_INSTALL_DIR}/qnx/libhexagon_controller.so")
else()
message(FATAL_ERROR "libhexagon_controller.so only support android and qnx")
endif()

add_library(hexagon_controller SHARED IMPORTED GLOBAL)
set_target_properties(hexagon_controller PROPERTIES IMPORTED_LOCATION ${NNLIB_CONTROLLER})

install(FILES ${NNLIB_CONTROLLER} DESTINATION lib)
