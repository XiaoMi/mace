#!/bin/bash

BUILD_DIR=build/micro/gcc-arm-none-eabi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../../micro \
  -DCMAKE_TOOLCHAIN_FILE=./cmake/toolchain/gcc-arm-none-eabi.cmake \
  -DMACE_MICRO_ENABLE_CMSIS=ON \
  -DCMAKE_INSTALL_PREFIX=install \
  -DMACE_MICRO_ENABLE_TESTS=OFF \
  $@ || exit 1

cmake --build . --target install -- -j || exit 1

cd ../../..
