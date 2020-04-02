Basic usage for Micro Controllers
==================================


Build and run an example model
-------------------------------

At first, make sure the environment has been set up correctly already (refer to :doc:`../installation/env_requirement`).

The followings are instructions about how to quickly build and run a provided model in
`MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__.

Here we use the har-cnn model as an example.

**Commands**

    1. Pull `MACE <https://github.com/XiaoMi/mace>`__ project.

    .. code-block:: sh

        git clone https://github.com/XiaoMi/mace.git
        cd mace/
        git fetch --all --tags --prune

        # Checkout the latest tag (i.e. release version)
        tag_name=`git describe --abbrev=0 --tags`
        git checkout tags/${tag_name}

    .. note::

        It's highly recommended to use a release version instead of master branch.


    2. Pull `MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__ project.

    .. code-block:: sh

        git clone https://github.com/XiaoMi/mace-models.git


    3. Convert the pre-trained har-cnn model to c++ code.

    .. code-block:: sh

        cd path/to/mace
        # output lib path: build/har-cnn/model/har_cnn_micro.tar.gz
        CONF_FILE=/path/to/mace-models/micro-models/har-cnn/har-cnn.yml
        python tools/converter.py convert --config=$CONF_FILE --enable_micro


    4. Build Micro-Controllers engine and models to library on host.

    .. code-block:: sh

        # copy convert result to micro dir ``path/to/micro``
        cp build/har-cnn/model/har_cnn_micro.tar.gz path/to/micro/
        cd path/to/micro
        tar zxvf har_cnn_micro.tar.gz
        bazel build //micro/codegen:micro_engine

    .. note::

        - This step can be skipped if you just want to run a model using ``tools/python/run_micro.py``, such as commands in step 5.

        - The build result ``bazel-bin/micro/codegen/libmicro_engine.so``'s abi is host, if you want to run the model on micro controllers, you should build the code with the target abi.

    5. Run the model on host.

    .. code-block:: sh

        CONF_FILE=/path/to/mace-models/micro-models/har-cnn/har-cnn.yml
        # Run
        python tools/python/run_micro.py --config $CONF_FILE --model_name har_cnn --build

    	# Test model run time
        python tools/python/run_micro.py --config $CONF_FILE --model_name har_cnn --build --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/python/run_micro.py --config $CONF_FILE --model_name har_cnn --build --validate
        # Validate the layers' correctness.
        python tools/python/run_micro.py --config $CONF_FILE --model_name har_cnn --build --validate --layers 0:-1



Deploy your model into applications
------------------------------------

Please refer to \ ``/mace/micro/tools/micro_run.cc`` for full usage. The following list the key steps.

.. code-block:: cpp

    // Include the headers
    #include "micro/include/public/micro.h"

    // 1. Create MaceMicroEngine instance
    MaceMicroEngine *micro_engine = nullptr;
    MaceStatus status = har_cnn::GetMicroEngineSingleton(&micro_engine);

    // 1. Create and register Input buffers
    std::vector<std::shared_ptr<char>> inputs;
    std::vector<int32_t> input_sizes;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      input_sizes.push_back(std::accumulate(input_shapes[i].begin(),
                                            input_shapes[i].end(), sizeof(float),
                                            std::multiplies<int32_t>()));
      inputs.push_back(std::shared_ptr<char>(new char[input_sizes[i]],
                                             std::default_delete<char[]>()));
    }
    // TODO: fill data into input buffers
    for (size_t i = 0; i < input_names.size(); ++i) {
      micro_engine->RegisterInputData(i, inputs[i].get(),
                                      input_shapes[i].data());
    }

    // 3. Run the model
    MaceStatus status = micro_engine->Run();

    // 4. Get the results
    for (size_t i = 0; i < output_names.size(); ++i) {
      void *output_buffer = nullptr;
      const int32_t *output_dims = nullptr;
      uint32_t dim_size = 0;
      MaceStatus status =
          micro_engine->GetOutputData(i, &output_buffer, &output_dims, &dim_size);
      // TODO: the result data is in output_buffer, you can not delete output_buffer.
    }
