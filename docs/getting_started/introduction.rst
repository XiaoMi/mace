Introduction
============

Mobile AI Compute Engine (MACE) is a deep learning inference framework optimized for
mobile heterogeneous computing platforms. The following figure shows the
overall architecture.

.. image:: mace-arch.png
   :scale: 40 %
   :align: center

Model format
------------

MACE defines a customized model format which is similar to
Caffe2. The MACE model can be converted from exported models by TensorFlow
and Caffe. A YAML file is used to describe the model deployment details. In the
next chapter, there is a detailed guide showing how to create this YAML file.

Model conversion
----------------

Currently, we provide model converters for TensorFlow and Caffe. And
more frameworks will be supported in the future.

Model loading
-------------

The MACE model format contains two parts: the model graph definition and
the model parameter tensors. The graph part utilizes Protocol Buffers
for serialization. All the model parameter tensors are concatenated
together into a continuous byte array, and we call this array tensor data in
the following paragraphs. In the model graph, the tensor data offsets
and lengths are recorded.

The models can be loaded in 3 ways:

1. Both model graph and tensor data are dynamically loaded externally
   (by default, from file system, but the users are free to choose their own
   implementations, for example, with compression or encryption). This
   approach provides the most flexibility but the weakest model protection.
2. Both model graph and tensor data are converted into C++ code and loaded
   by executing the compiled code. This approach provides the strongest
   model protection and simplest deployment.
3. The model graph is converted into C++ code and constructed as the second
   approach, and the tensor data is loaded externally as the first approach.
