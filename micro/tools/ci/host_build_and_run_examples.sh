#! /bin/bash

python tools/python/convert.py --config micro/pretrained_models/tensorflow/kws/kws-tc_resnet8.yml --enable_micro || exit -1

rm -rf build/micro
./micro/tools/cmake/cmake-build-host.sh \
-DMACE_MICRO_ENABLE_EXAMPLES=ON  -DMICRO_MODEL_NAME=kws_tc_resnet8 -DMICRO_DATA_NAME=kws \
-DMACE_MICRO_ENABLE_TESTS=OFF \
-DMACE_MICRO_ENABLE_CMSIS=OFF || exit -1

./build/micro/host/examples/classifier/kws_tc_resnet8

python3 tools/python/convert.py --config micro/pretrained_models/keras/mnist/mnist-int8.yml --enable_micro || exit -1

rm -rf build/micro
./micro/tools/cmake/cmake-build-host.sh \
-DMACE_MICRO_ENABLE_CMSIS=ON \
-DMACE_MICRO_ENABLE_EXAMPLES=ON \
-DMICRO_MODEL_NAME=mnist_int8 -DMICRO_DATA_NAME=mnist \
-DMACE_MICRO_ENABLE_TESTS=OFF || exit -1

./build/micro/host/examples/classifier/mnist_int8

cd ..