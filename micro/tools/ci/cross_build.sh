#! /bin/bash

echo "Builds host float32"
rm -rf build/micro
./micro/tools/cmake/cmake-build-host.sh \
-DMACE_MICRO_ENABLE_TESTS=ON \
-DMACE_MICRO_ENABLE_CMSIS=ON || exit -1

echo "Builds host bfloat16"
rm -rf build/micro
./micro/tools/cmake/cmake-build-host.sh \
-DMACE_MICRO_ENABLE_BFLOAT16=ON \
-DMACE_MICRO_ENABLE_TESTS=ON \
-DMACE_MICRO_ENABLE_CMSIS=ON || exit -1

echo "Builds gcc arm cortex-m7"
rm -rf build/micro
./micro/tools/cmake/cmake-build-gcc-arm-none-eabi.sh \
-DARM_CPU=cortex-m7 \
-DMACE_MICRO_ENABLE_TESTS=OFF \
-DMACE_MICRO_ENABLE_CMSIS=ON  || exit -1

cd ..