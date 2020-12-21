Basic usage for Micro Controllers
==================================

MACE Micro is a lightweight neural network inference engine for MCUs and low-power DSPs.
At now we support Cortex-M MCUs and Qualcomm Hexagon DSPs. You can get our projects from GitHub.

Get MACE Micro Projects
-----------------------

MACE Micro is a sub project of MACE, so you can get it from MACE.

.. code-block:: sh

    git clone https://github.com/XiaoMi/mace.git
    # Inits submodules by yourself
    cd mace && git submodule update --init micro && cd ..

Environment Requirements
------------------------

On a ubuntu18.04/20.04 PC, do the following steps.

.. code-block:: sh

    apt-get update
    apt-get install -y wget

    apt-get install -y g++
    # Required for Cortex-M MCUs
    apt-get install -y gcc-arm-none-eabi
    apt-get install -y python3 python3-pip

    python3 -m pip install jinja2 pyyaml sh numpy six filelock
    # Installs cmake above 3.13.0
    wget https://cnbj1.fds.api.xiaomi.com/mace/third-party/cmake-3.18.3-Linux-x86_64.sh
    chmod +x cmake-3.18.3-Linux-x86_64.sh && ./cmake-3.18.3-Linux-x86_64.sh --skip-license --prefix=/usr

    python3 -m pip install -U pip
    # The Tensorflow version depends on your model
    # The Tensroflow 1.x frozen model and Tensorflow 2.x Keras model are both supported
    python3 -m pip install tensorflow==2.3.0
    python3 -m pip install tensorflow_model_optimization

You also can use a docker as the environment.

.. code-block:: sh

    cd mace/docker/mace-micro-dev
    docker build . -f mace-micro-dev.dockerfile --tag mace-micro-dev
    cd ../../..
    # Maps your workspace to docker container
    docker run -ti -v $(pwd):/workspace/ -w /workspace  mace-micro-dev


Convert a model to c++ code
----------------------------

Here we use a pre-trained model of the MNIST database,

.. code-block:: sh

    cd mace
    # Converts a tensorflow 2.x keras model, you need install python3 and tensorflow==2.x additional
    python3 tools/python/convert.py --config=micro/pretrained_models/keras/mnist/mnist.yml --enable_micro


Model config file
-----------------

The following is a completed model config file,

.. code-block:: sh

    library_name: mnist
    target_abis: [host]
    model_graph_format: file
    model_data_format: file
    models:
      mnist_int8:
        platform: keras
        model_file_path: https://cnbj1.fds.api.xiaomi.com/mace/miai-models/micro/keras/mnist/mnist-int8.h5
        model_sha256_checksum: 0ff90446134c41fb5e0524484cd9d7452282d3825f13b839c364a58abd0490ee
        subgraphs:
          - input_tensors:
              - conv2d_input:0
            input_shapes:
              - 1,28,28,1
            input_ranges:
              - 0,1
            output_tensors:
              - quant_dense_1/Softmax:0
            output_shapes:
              - 1,10
            validation_inputs_data:
              - https://cnbj1.fds.api.xiaomi.com/mace/inputs/mnist4.npy
        runtime: cpu
        quantize: 1
        quantize_schema: int8
        micro:
          backend: cmsis # Micro will use CMSIS_5 NN modules

For the bfloat16 model,

.. code-block:: yaml

    data_type: bf16_fp32

For the int8 model,

.. code-block:: yaml

    quantize: 1
    quantize_schema: int8
    # Required when your model has not quantize info
    quantize_range_file: range_file_path



Build MACE Micro and models libraries
--------------------------------------

Here, we build the MACE Micro engine and models to libraries on a linux host machine. The CMake build parameters depends on your model config file.

For float32 model,

.. code-block:: sh

    ./micro/tools/cmake/cmake-build-host.sh

For bfloat16 model,

.. code-block:: sh

    ./micro/tools/cmake/cmake-build-host.sh -DMACE_MICRO_ENABLE_BFLOAT16=ON

.. note::

    You can only use either float32 or bfloat16

For int8 model,

.. code-block:: sh

    ./micro/tools/cmake/cmake-build-host.sh -DMACE_MICRO_ENABLE_CMSIS=ON

Use libraries directly
-----------------------

With these steps, we can find necessary libraries and headers in the "build/micro/host/install" directory, you can use the libraries directly.

.. code-block:: sh

    # Builds example
    g++ micro/examples/classifier/main.cc -DMICRO_MODEL_NAME=mnist -DMICRO_DATA_NAME=mnist  -I build/micro/host/install/include/ -L build/micro/host/install/lib/ -lmicro  -lmodels -lmicro -o mnist
    # Runs the mnist example
    ./mnist


Code example
------------------------------------

The following code is the mnist example source files, which the main steps is annotated

.. code-block:: cpp

    #include "data/mnist.h"

    #include <cstdio>

    // Include MACE Micro header
    #include "micro.h"

    namespace micro {
    namespace mnist {

    // We use forward declaration to avoid include the special engine header
    MaceStatus GetMicroEngineSingleton(MaceMicroEngine **engine);

    }
    }  // namespace micro

    int main() {
      // Step 1, get the mnist micro engine
      micro::MaceMicroEngine *micro_engine = NULL;
      micro::MaceStatus status =
          micro::mnist::GetMicroEngineSingleton(&micro_engine);

      // Step 2, set input data
      static float *input_data = data_mnist_4;
      int32_t input_dims[4] = {1, 28, 28, 1};
      micro_engine->RegisterInputData(0, input_data, input_dims);

      // Step3, run the inference
      micro_engine->Run();

      // Step 4, get output data
      float *output_buffer = NULL;
      const int32_t *output_dims = NULL;
      uint32_t dim_size = 0;
      micro_engine->GetOutputData(
          0, reinterpret_cast<void **>(&output_buffer), &output_dims, &dim_size);

      for (int32_t i = 0; i < output_dims[1]; ++i) {
        printf("%d: %f\n", i, output_buffer[i]);
      }

      return 0;
    }

For more examples, goto the directory "micro/examples"

Performance
-----------

We deploy a `HAR-CNN <https://github.com/Shahnawax/HAR-CNN-Keras>`__ int8 model on the NUCLEO-F767ZI(Cortex-M7) board. Each inference of HAR CNN model takes 12 ms.