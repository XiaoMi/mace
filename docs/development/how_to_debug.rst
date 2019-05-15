How to debug
==========================

Debug correctness
--------------------------

MACE provides tools to examine correctness of model execution by comparing model's output of MACE with output of training platform (e.g., Tensorflow, Caffe).
Three metrics are used as comparison results:

* **Cosine Similarity**:

.. math::

	Cosine\ Similarity = \frac{X \cdot X'}{\|X\| \|X'\|}

This metric will be approximately equal to 1 if output is correct.

* **SQNR** (Signal-to-Quantization-Noise Ratio):

.. math::

	SQNR = \frac{P_{signal}}{P_{noise}} = \frac{\|X\|^2}{\|X - X'\|^2}

It is usually used to measure quantization accuracy. The higher SQNR is, the better accuracy will be.

* **Pixel Accuracy**:

.. math::

   Pixel\ Accuracy = \frac{\sum^{batch}_{b=1} equal(\mathrm{argmax} X_b, \mathrm{argmax} X'_b)}{batch}

It is usually used to measure classification accuracy. The higher the better.

where :math:`X` is expected output (from training platform) whereas :math:`X'` is actual output (from MACE) .


You can validate it by specifying `--validate` while running the model.

    .. code:: sh

        # Validate the correctness by comparing the results against the
        # original model and framework
        python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --validate

MACE automatically validate these metrics by running models with synthetic inputs.
If you want to specify input data to use, you can add an option in yaml config under 'subgraphs', e.g.,

.. code:: yaml

	models:
	  mobilenet_v1:
	    platform: tensorflow
	    model_file_path: https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb
	    model_sha256_checksum: 71b10f540ece33c49a7b51f5d4095fc9bd78ce46ebf0300487b2ee23d71294e6
	    subgraphs:
	      - input_tensors:
	          - input
	        input_shapes:
	          - 1,224,224,3
	        output_tensors:
	          - MobilenetV1/Predictions/Reshape_1
	        output_shapes:
	          - 1,1001
	        check_tensors:
	          - MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd:0
	        check_shapes:
	          - 1,1,1,1001
	        validation_inputs_data:
	          - https://cnbj1.fds.api.xiaomi.com/mace/inputs/dog.npy

If model's output is suspected to be incorrect, it might be useful to debug your model layer by layer by specifying an intermediate layer as output,
or use binary search method until suspicious layer is found.

You can also specify `--layers` after `--validate` to validate all or some of the layers of the model(excluding some layers changed by MACE, e.g., BatchToSpaceND),
it only supports TensorFlow now. You can find validation results in `build/your_model/model/runtime_in_yaml/log.csv`.

For quantized model, if you want to check one layer, you can add `check_tensors` and `check_shapes` like in the yaml above. You can only specify
MACE op's output.


Debug with crash
--------------------------
When MACE crashes, a complete stacktrace is useful in debugging. But because of `selinux problem <https://github.com/android-ndk/ndk/issues/943#issuecomment-477834810>`__,
symbols table is not loaded in memory, which leading to no symbol in stack trace.
To circumvent this problem, you can rebuild `mace_run` with `--debug_mode` option to reserve debug symbols, e.g.,

  .. code:: sh

    python tools/converter.py run --config=/path/to/config.yml --debug_mode

For android, you can use `ndk-stack tools <https://developer.android.com/ndk/guides/ndk-stack?hl=EN>`__ to symbolize stack trace, e.g.,

  .. code:: sh

    adb logcat | $ANDROID_NDK_HOME/ndk-stack -sym /path/to/local/binary/directory/


Debug memory usage
--------------------------
The simplest way to debug process memory usage is to use ``top`` command. With ``-H`` option, it can also show thread info.
For android, if you need more memory info, e.g., memory used of all categories, ``adb shell dumpsys meminfo`` will help.
By watching memory usage, you can check if memory usage meets expectations or if any leak happens.


Debug performance
--------------------------
Using MACE, you can benchmark a model by examining each layer's duration as well as total duration. Or you can benchmark a single op.
The detailed information is in :doc:`../user_guide/benchmark`.


Debug model conversion
--------------------------
After model is converted to MACE model, a literal model graph is generated in directory `mace/codegen/models/your_model`.
You can refer to it when debugging model conversion.

MACE also provides model visualization HTML generated in `build` directory, generated after converting model.


Debug engine using log
--------------------------
MACE implements a similar logging mechanism like `glog <https://github.com/google/glog>`__.
There are two types of logs, LOG for normal logging and VLOG for debugging.

LOG includes four levels, sorted by severity level: ``INFO``, ``WARNING``, ``ERROR``, ``FATAL``.
The logging severity threshold can be configured via environment variable, e.g. ``MACE_CPP_MIN_LOG_LEVEL=WARNING`` to set as ``WARNING``.
Only the log messages with equal or above the specified severity threshold will be printed, the default threshold is ``INFO``.
We don't support integer log severity value like `glog <https://github.com/google/glog>`__, because they are confusing with VLOG.

VLOG is verbose logging which is logged as ``LOG(INFO)``. VLOG also has more detailed integer verbose levels, like 0, 1, 2, 3, etc.
The threshold can be configured through environment variable, e.g. ``MACE_CPP_MIN_VLOG_LEVEL=2`` to set as ``2``.
With VLOG, the lower the verbose level, the more likely messages are to be logged. For example, when the threshold is set
to 2, both ``VLOG(1)``, ``VLOG(2)`` log messages will be printed, but ``VLOG(3)`` and highers won't. 

By using ``mace_run`` tool, VLOG level can be easily set by option, e.g.,

	.. code:: sh

		python tools/converter.py run --config /path/to/model.yml --vlog_level=2


If models are run on android, you might need to use ``adb logcat`` to view logs.


Debug engine using GDB
--------------------------
GDB can be used as the last resort, as it is powerful that it can trace stacks of your process. If you run models on android,
things may be a little bit complicated.

	.. code:: sh

		# push gdbserver to your phone
		adb push $ANDROID_NDK_HOME/prebuilt/android-arm64/gdbserver/gdbserver /data/local/tmp/


		# set system env, pull system libs and bins to host
		export SYSTEM_LIB=/path/to/android/system_lib
		export SYSTEM_BIN=/path/to/android/system_bin
		mkdir -p $SYSTEM_LIB
		adb pull /system/lib/. $SYSTEM_LIB
		mkdir -p $SYSTEM_BIN
		adb pull /system/bin/. $SYSTEM_BIN


		# Suppose ndk compiler used to compile Mace is of android-21
		export PLATFORMS_21_LIB=$ANDROID_NDK_HOME/platforms/android-21/arch-arm/usr/lib/


		# start gdbserver，make gdb listen to port 6000
		# adb shell /data/local/tmp/gdbserver :6000 /path/to/binary/on/phone/example_bin
		adb shell LD_LIBRARY_PATH=/dir/to/dynamic/library/on/phone/ /data/local/tmp/gdbserver :6000 /data/local/tmp/mace_run/example_bin
		# or attach a running process
		adb shell /data/local/tmp/gdbserver :6000 --attach 8700
		# forward tcp port
		adb forward tcp:6000 tcp:6000


		# use gdb on host to execute binary
		$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/gdb [/path/to/binary/on/host/example_bin]


		# connect remote port after starting gdb command
		target remote :6000


		# set lib path
		set solib-search-path $SYSTEM_LIB:$SYSTEM_BIN:$PLATFORMS_21_LIB

		# then you can use it as host gdb, e.g.,
		bt

