#!/bin/bash
if [ -z "$GCC_ARM_ROOT" ]; then
  echo "GCC_ARM_ROOT is undefined";
fi

BUILD_DIR=build/cmake-build/gcc-arm-none-eabi
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../.. \
  -DGCC_ARM_ROOT=${GCC_ARM_ROOT} \
  -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchain/gcc-arm-none-eabi.cmake \
  -DCMAKE_INSTALL_PREFIX=install \
  $@ || exit 1

cmake --build . -- -j || exit 1

cd ../../..
