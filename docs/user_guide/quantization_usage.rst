Quantization
===============

MACE supports two kinds of quantization mechanisms, i.e.,

* **Quantization-aware training (Recommend)**

After pre-training model using float point, insert simulated quantization operations into the model. Fine tune the new model.
Refer to `Tensorflow quantization-aware training <https://github.com/tensorflow/tensorflow/tree/r1.15/tensorflow/contrib/quantize>`__.

* **Post-training quantization**

After pre-training model using float point, estimate output range of each activation layer using sample inputs.


.. note::

  `quantize_weights` and `quantize_nodes` should not be specified when using `TransformGraph` tool if using MACE quantization.


Quantization-aware training
----------------------------
It is recommended that developers fine tune the fixed-point model, as experiments show that by this way accuracy could be improved, especially for lightweight
models, e.g., MobileNet. The only thing you need to make it run using MACE is to add the following config to model yaml file:

  1. `input_ranges`: the ranges of model's inputs, e.g., -1.0,1.0.

  2. `quantize`: set `quantize` to be 1.


Post-training quantization
---------------------------
MACE supports post-training quantization if you want to take a chance to quantize model directly without fine tuning.
This method requires developer to calculate tensor range of each activation layer statistically using sample inputs.
MACE provides tools to do statistics with following steps (using `inception-v3` from `MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__ as an example):

  1. Convert original model to run on CPU host without obfuscation (by setting `target_abis` to `host`, `runtime` to `cpu`,
  and `obfuscate` to `0`, appending `:0` to `output_tensors` if missing in yaml config).

  .. code-block:: sh

    # For CMake users:
    python tools/python/convert.py --config ../mace-models/inception-v3/inception-v3.yml
      --quantize_stat

    # For Bazel users:
    python tools/converter.py convert --config ../mace-models/inception-v3/inception-v3.yml
      --quantize_stat

  2. Log tensor range of each activation layer by inferring several samples on CPU host. Sample inputs should be
  representative to calculate the ranges of each layer properly.

  .. code-block:: sh

    # Convert images to input tensors for MACE, see image_to_tensor.py for more arguments.
    python tools/image/image_to_tensor.py --input /path/to/directory/of/input/images
      --output_dir /path/to/directory/of/input/tensors --image_shape=299,299,3

    # Rename input tensors to start with input tensor name(to differentiate multiple
    # inputs of a model), input tensor name is what you specified as "input_tensors"
    # in yaml config. For example, "input" is the input tensor name of InceptionV3 as below.
    rename 's/^/input/' *

    # Run with input tensors
    # For CMake users:
    python tools/python/run_model.py --config ../mace-models/inception-v3/inception-v3.yml
      --quantize_stat --input_dir /path/to/directory/of/input/tensors > range_log

    # For Bazel users:
    python tools/converter.py run --config ../mace-models/inception-v3/inception-v3.yml
      --quantize_stat --input_dir /path/to/directory/of/input/tensors > range_log


  3. Calculate overall range of each activation layer. You may specify `--percentile` or `--enhance` and `--enhance_ratio`
  to try different ranges and see which is better. Experimentation shows that the default `percentile` and `enhance_ratio`
  works fine for several common models.

  .. code-block:: sh

    python tools/python/quantize/quantize_stat.py --log_file range_log > overall_range


  4. Convert quantized model (by setting `target_abis` to the final target abis, e.g., `armeabi-v7a`,
  `quantize` to `1` and `quantize_range_file` to the overall_range file path in yaml config).


Mixing usage
---------------------------
As `quantization-aware training` is still evolving, there are some operations that are not supported,
which leaves some activation layers without tensor range. In this case, `post-training quantization`
can be used to calculate these missing ranges. To mix the usage, just get a `quantization-aware training`
model and then go through all the steps of `post-training quantization`. MACE will use the tensor ranges
from the `overall_range` file of `post-training quantization` if the ranges are missing from the
`quantization-aware training` model.


Supported devices
-----------------
MACE supports running quantized models on ARM CPU and other acceleration devices, e.g., Qualcomm Hexagon DSP, MediaTek APU.
ARM CPU is ubiquitous, which can speed up most of edge devices. However, AI specialized devices may run much faster
than ARM CPU, and in the meantime consume much lower power. Headers and libraries of these devices can be found in `third_party`
directory.

* To run models on **ARM CPU**, users should

  1. Set `runtime` in yaml config to `cpu` (`Armv8.2+dotproduct` instructions will be used automatically
     if detected by `getauxval`, which can greatly improve convolution/gemm performance).
  
* To run models on **Hexagon DSP**, users should

  1. Set `runtime` in yaml config to `dsp`.

  2. Make sure SOCs of the phone is manufactured by Qualcomm and has HVX supported.

  3. Make sure the phone disables secure boot (once enabled, cannot be reversed, so you probably can only get that type
     phones from manufacturers). This can be checked by executing the following command.

   .. code-block:: sh

       adb shell getprop ro.boot.secureboot

   The return value should be 0.

  4. Root the phone.

  5. Sign the phone by using testsig provided by Qualcomm. (Download Qualcomm Hexagon SDK first, plugin the phone to PC,
     run scripts/testsig.py)

  6. Push `third_party/nnlib/v6x/libhexagon_nn_skel.so` to `/system/vendor/lib/rfsa/adsp/`. You can check
     `docs/feature_matrix.html` in Hexagon SDK to make sure which version to use.

Then, there you go, you can run Mace on Hexagon DSP. This indeed seems like a whole lot of work to do. Well, the good news
is that starting in the SM8150 family(some devices with old firmware may still not work), signature-free dynamic
module offload is enabled on cDSP. So, steps 2-4 can be skipped. This can be achieved by calling `SetHexagonToUnsignedPD()`
before creating MACE engine.

* To run models on **MediaTek APU**, users should

  1. Set `runtime` in yaml config to `apu`.

  2. Make sure SOCs of the phone is manufactured by MediaTek and has APU supported.

  3. Push `third_party/apu/mtxxxx/libapu-platform.so` to `/vendor/lib64/`.
