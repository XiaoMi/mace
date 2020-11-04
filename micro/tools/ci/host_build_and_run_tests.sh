#! /bin/bash

rm -rf build/micro
./micro/tools/cmake/cmake-build-host.sh \
-DMACE_MICRO_ENABLE_TESTS=ON \
-DMACE_MICRO_ENABLE_CMSIS=ON || exit -1

echo "MACE Micro ut"
build/micro/host/test/ccunit/micro_ops_test || exit -1

echo "MACE Micro benchmark"
build/micro/host/test/ccbenchmark/micro_cc_benchmark || exit -1

cd ..