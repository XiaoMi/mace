INCLUDE(ExternalProject)

set(GFLAGS_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/gflags")
set(GFLAGS_INSTALL_DIR "${MACE_THIRD_PARTY_DIR}/install/gflags")
set(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)

if(MSVC)
  set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/gflags_static.lib" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
else(MSVC)
  set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
endif(MSVC)

include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})

# Mirror of https://github.com/gflags/gflags/archive/v2.2.2.zip
set(GFLAGS_URL     "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gflags/v2.2.2.zip")
set(GFLAGS_HASH    "SHA256=19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5")

ExternalProject_Add(
  gflags_gflags
  URL_HASH         "${GFLAGS_HASH}"
  URL              "${GFLAGS_URL}"
  PREFIX           ${GFLAGS_SRCS_DIR}
  UPDATE_COMMAND   ""
  BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES}
  CMAKE_ARGS       -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
                   -DBUILD_STATIC_LIBS=ON
                   -DBUILD_TESTING=OFF
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                   ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
)

if(MSVC)
  add_custom_command(TARGET gflags_gflags POST_BUILD
    COMMAND if $<CONFIG:Debug>==1 (${CMAKE_COMMAND} -E copy ${GFLAGS_INSTALL_DIR}/lib/gflags_static_debug.lib ${GFLAGS_INSTALL_DIR}/lib/gflags_static.lib)
  )
endif(MSVC)

add_library(gflags STATIC IMPORTED GLOBAL)
set_property(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES})
add_dependencies(gflags gflags_gflags)

if(MSVC)
  set_target_properties(gflags
    PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES
    Shlwapi.lib)
endif(MSVC)
