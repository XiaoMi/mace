Dynamic LSTM
============


The DynamicLSTM in MACE is implemented for Kaldi's time delay RNN models.

The following pictures explain how to fuse components into a DynamicLSTMCell.

Before fusing:

.. image:: imgs/FuseLSTM.png
   :scale: 100 %
   :align: center


After fusing:

.. image:: imgs/DynamicLSTM.png
   :scale: 100 %
   :align: center

For more details about LSTMNonlinear in Kaldi,
please refer to [LstmNonlinearComponent](http://kaldi-asr.org/doc/nnet-simple-component_8h_source.html#l02164)