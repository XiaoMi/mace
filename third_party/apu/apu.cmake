set(APU_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/apu")
set(APU_INCLUDE_DIR  "${APU_INSTALL_DIR}")

include_directories(SYSTEM "${APU_INCLUDE_DIR}")

set(APU-FRONTEND
  "${APU_INSTALL_DIR}/libapu-frontend.so"
)

add_library(apu-frontend SHARED IMPORTED GLOBAL)
set_target_properties(apu-frontend PROPERTIES IMPORTED_LOCATION ${APU-FRONTEND})

install(FILES ${APU-FRONTEND} DESTINATION lib)
