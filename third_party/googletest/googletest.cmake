if(MACE_ENABLE_TESTS)
  enable_testing()

  include(ExternalProject)

  set(GTEST_SOURCES_DIR ${MACE_THIRD_PARTY_DIR}/gtest)
  set(GTEST_INSTALL_DIR ${MACE_THIRD_PARTY_DIR}/install/gtest)
  set(GTEST_INCLUDE_DIR "${GTEST_INSTALL_DIR}/include" CACHE PATH "gtest include directory." FORCE)

  include_directories(SYSTEM ${GTEST_INCLUDE_DIR})

  if(MSVC)
    set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/gtest.lib" CACHE FILEPATH "gtest libraries." FORCE)
    set(GTEST_MAIN_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/gtest_main.lib" CACHE FILEPATH "gtest main libraries." FORCE)
  else(MSVC)
    set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/libgtest.a" CACHE FILEPATH "gtest libraries." FORCE)
    set(GTEST_MAIN_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/libgtest_main.a" CACHE FILEPATH "gtest main libraries." FORCE)
  endif(MSVC)

  # Mirror of "https://github.com/google/googletest/archive/release-1.8.0.zip"
  set(GTEST_URL  "https://cnbj1.fds.api.xiaomi.com/mace/third-party/googletest/googletest-release-1.8.0.zip")
  set(GTEST_HASH "SHA256=f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf")

  ExternalProject_Add(
    extern_gtest
    URL_HASH         "${GTEST_HASH}"
    URL              "${GTEST_URL}"
    PREFIX           ${GTEST_SOURCES_DIR}
    UPDATE_COMMAND   ""
    BUILD_BYPRODUCTS ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES}
    CMAKE_ARGS       -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
                     -DBUILD_GMOCK=ON
                     -Dgtest_disable_pthreads=ON
                     -Dgtest_force_shared_crt=ON
                     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                     -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                     ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
  )

  add_library(gtest STATIC IMPORTED GLOBAL)
  set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES})
  add_dependencies(gtest extern_gtest)

  add_library(gtest_main STATIC IMPORTED GLOBAL)
  set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${GTEST_MAIN_LIBRARIES})
  add_dependencies(gtest_main extern_gtest)

endif(MACE_ENABLE_TESTS)
