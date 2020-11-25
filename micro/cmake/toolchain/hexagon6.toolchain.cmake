set(CMAKE_C_COMPILE_WORKS TRUE)
set(CMAKE_CXX_COMPILE_WORKS TRUE)

set(CMAKE_C_COMPILER "${HEXAGON_TOOLS}/qc/bin/hexagon-clang")
set(CMAKE_CXX_COMPILER "${HEXAGON_TOOLS}/qc/bin/hexagon-clang")
set(CMAKE_AR "${HEXAGON_TOOLS}/gnu/bin/hexagon-ar" CACHE FILEPATH "Archiver")
set(CMAKE_LINKER "${HEXAGON_TOOLS}/qc/bin/hexagon-llvm-link")

add_compile_options(-Wall -Wpointer-arith -Wstrict-prototypes -Wnested-externs -mv55 -Uqdsp6 -Uq6sim -Uqdsp6r0 -Os -g -fdata-sections -ffunction-sections -nostdlib -fno-exceptions -fno-strict-aliasing -fstack-protector -Wno-low -mllvm -disable-store-widen -mllvm -Werror -Wall -Wextra -Wno-sign-compare -Wno-unused-result -Wno-unused-parameter)

set(CMAKE_FIND_ROOT_PATH "${HEXAGON_TOOLS}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
add_definitions("-D_HAS_C9X")

set(HEXAGON ON)
set(HEXAGON6 ON)
