Dynamic LSTM
==================


The DynamicLSTM in MACE is implemented for Kaldi's time delay RNN models.

The following pictures explain how to fuse components into a DynamicLSTMCell.

Before fusing:

<div  align="left">
<img src="imgs/FuseLSTM.png" width = "320" height = "960" alt="how to fuse lstm" />
</div>


After fusing:

<div  align="left">
<img src="imgs/DynamicLSTM.png" width = "358" height = "391" alt="DynamicLSTM" />
</div>


For more details about LSTMNonlinear in Kaldi,
please refer to [LstmNonlinearComponent](http://kaldi-asr.org/doc/nnet-simple-component_8h_source.html#l02164)