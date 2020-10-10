
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_FIND_ROOT_PATH "${GCC_ARM_ROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

find_program(CMAKE_C_COMPILER NAMES arm-none-eabi-gcc arm-none-eabi-gcc.exe)
find_program(CMAKE_CXX_COMPILER NAMES arm-none-eabi-g++ arm-none-eabi-g++.exe)
find_program(CMAKE_ASM_COMPILER NAMES arm-none-eabi-gcc arm-none-eabi-gcc.exe)
find_program(CMAKE_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar.exe)
find_program(CMAKE_CXX_COMPILER_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar.exe)
find_program(CMAKE_C_COMPILER_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar.exe)
find_program(CMAKE_LINKER NAMES arm-none-eabi-g++ arm-none-eabi-g++.exe)

find_program(ELF2BIN NAMES arm-none-eabi-objcopy arm-none-eabi-objcopy.exe)

# Force compiler settings
SET(CMAKE_C_COMPILER_WORKS TRUE)
SET(CMAKE_CXX_COMPILER_WORKS TRUE)

set(MACE_MICRO_GCC_ARM ON)
