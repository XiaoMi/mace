#!/usr/bin/env sh

set -e

# build for host
BUILD_DIR=build/host
rm -rf ${BUILD_DIR} && mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DMACE_ENABLE_NEON=OFF         \
      -DMACE_ENABLE_QUANTIZE=OFF     \
      -DMACE_ENABLE_OPENCL=ON        \
      -DCMAKE_INSTALL_PREFIX=install \
      ../..
make -j8 VERBOSE=1 && make install
cd ../..
