Quantization
===============

MACE supports two kinds of quantization mechanisms, i.e.,

* **Quantization-aware training (Recommend)**

After pre-training model using float point, insert simulated quantization operations into the model. Fine tune the new model.
Refer to `Tensorflow quantization-aware training <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize>`__.

* **Post training quantization**

After pre-training model using float point, estimate output range of each activation layer using sample inputs.


Quantization-aware training
----------------------------
It is recommended that developers fine tune the fixed-point model, as experiments show that by this way accuracy could be improved, especially for lightweight
models, e.g., MobileNet. The only thing you need to make it run using MACE is to add the following config to model yaml file:

	1. `input_ranges`: the ranges of model's inputs, e.g., -1.0,1.0.

	2. `quantize`: set `quantize` to be 1.


Post training quantization
---------------------------
MACE supports post-training quantization if you want to take a chance to quantize model directly without fine tuning.
This method requires developer to calculate tensor range of each activation layer statistically using sample inputs.
MACE provides tools to do statistics with following steps:

	1. Convert original model to run on CPU host without obfuscation (by setting `target_abis` to `host`, `runtime` to `cpu`, and `obfuscate` to `0`, appending `:0` to `output_tensors` if missing in yaml config).
	E.g.,

	.. code:: sh

		python tools/converter.py convert --config ../mace-models/inception-v3/inception-v3.yml


	2. Log tensor range of each activation layer by inferring several samples.

	.. code:: sh

		python tools/converter.py run --config ../mace-models/inception-v3/inception-v3.yml --example --quantize_stat --input_dir samples > range_log


	3. Calculate overall range of each activation layer by specifying percentage cutoff.

	.. code:: sh

		python mace/python/tools/quantization/quantize_stat.py --log_file range_log --percentile 5 > overall_range


	4. Convert quantized model (by setting `quantize` to `1` and `quantize_range_file` to the overall_range file path in yaml config).


.. note::

	`quantize_weights` and `quantize_nodes` should not be specified when using `TransformGraph` tool if using MACE quantization.
