include(ExternalProject)

set(GEMMLOWP_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/gemmlowp")
set(GEMMLOWP_INCLUDE_DIR "${GEMMLOWP_SRCS_DIR}/src/gemmlowp" CACHE PATH "gemmlowp include directory." FORCE)

include_directories(SYSTEM ${GEMMLOWP_INCLUDE_DIR})

set(GEMMLOWP_URL  "http://cnbj1.fds.api.xiaomi.com/mace/third-party/gemmlowp/gemmlowp-master-48c0547a046d49b466aa01e3a82a18028f288924.zip")
set(GEMMLOWP_HASH "SHA256=f340384e7728cea605e83597593699dfe8d13ff333b834d24c256935e3dc1758")

ExternalProject_Add(
  gemmlowp
  URL_HASH          "${GEMMLOWP_HASH}"
  URL               "${GEMMLOWP_URL}"
  PREFIX            "${GEMMLOWP_SRCS_DIR}"
  BINARY_DIR        ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
