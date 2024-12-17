set -ex

MACE_ROOT_DIR=$(pwd)

# build for arm linux aarch64
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR=build/cmake-build/qnx
fi

if [[ -z "$QNX_BIN_DIR" ]]; then
    echo "please set \$QNX_BIN_DIR as your qnx toolchain root"
    exit 1
fi

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake -DQNX=True \
      -DCROSSTOOL_ROOT=${QNX_BIN_DIR} \
      -DCMAKE_TOOLCHAIN_FILE=${MACE_ROOT_DIR}/cmake/toolchains/qnx.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DMACE_ENABLE_NEON=ON     \
      -DMACE_ENABLE_QUANTIZE=ON   \
      -DMACE_ENABLE_OPENCL=OFF       \
      -DMACE_ENABLE_CPU=ON                \
      -DMACE_ENABLE_BFLOAT16=OFF     \
      -DMACE_ENABLE_TESTS=OFF         \
      -DMACE_ENABLE_BENCHMARKS=OFF    \
      -DMACE_ENABLE_CODE_MODE=OFF    \
      -DMACE_ENABLE_HEXAGON_DSP=ON    \
      -DMACE_ENABLE_RPCMEM=ON    \
      -DCMAKE_INSTALL_PREFIX=install \
      ../../..

make $@ && make install
cd ../../..
