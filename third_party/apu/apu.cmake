set(APU_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/apu")
set(APU_INCLUDE_DIR  "${APU_INSTALL_DIR}")

include_directories(SYSTEM "${APU_INCLUDE_DIR}")

if(MACE_MTK_APU_ANCIENT)
  set(APU-FRONTEND "${APU_INSTALL_DIR}/android_Q/mt67xx/libapu-frontend.so")
else(MACE_MTK_APU_ANCIENT)  # for mt68xx on android Q it is only a place holder
  set(APU-FRONTEND "${APU_INSTALL_DIR}/android_R/libapu-frontend.so")
endif(MACE_MTK_APU_ANCIENT)

add_library(apu-frontend SHARED IMPORTED GLOBAL)
set_target_properties(apu-frontend PROPERTIES IMPORTED_LOCATION ${APU-FRONTEND})

install(FILES ${APU-FRONTEND} DESTINATION lib)
