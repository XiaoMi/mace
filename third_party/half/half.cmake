include(ExternalProject)

set(HALF_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/half")
set(HALF_INCLUDE_DIR "${HALF_SRCS_DIR}/src/half" CACHE PATH "half include directory." FORCE)

include_directories(SYSTEM ${HALF_INCLUDE_DIR})

set(HALF_URL  "https://cnbj1.fds.api.xiaomi.com/mace/third-party/half/half-code-356-trunk.zip")
set(HALF_HASH "SHA256=0f514a1e877932b21dc5edc26a148ddc700b6af2facfed4c030ca72f74d0219e")

ExternalProject_Add(
  half 
  URL               "${HALF_URL}"
  URL_HASH          "${HALF_HASH}"
  PREFIX            "${HALF_SRCS_DIR}"
  BINARY_DIR        ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
