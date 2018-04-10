import tensorflow as tf
from mace.proto import mace_pb2

TF_DTYPE_2_MACE_DTYPE_MAP = {
    tf.float32: mace_pb2.DT_FLOAT,
    tf.double: mace_pb2.DT_DOUBLE,
    tf.half: mace_pb2.DT_HALF,
    tf.int64: mace_pb2.DT_INT64,
    tf.int32: mace_pb2.DT_INT32,
    tf.qint32: mace_pb2.DT_INT32,
    tf.int16: mace_pb2.DT_INT16,
    tf.qint16: mace_pb2.DT_INT16,
    tf.int8: mace_pb2.DT_INT8,
    tf.qint8: mace_pb2.DT_INT8,
    tf.quint16: mace_pb2.DT_UINT16,
    tf.uint16: mace_pb2.DT_UINT16,
    tf.quint8: mace_pb2.DT_UINT8,
    tf.uint8: mace_pb2.DT_UINT8,
    tf.string: mace_pb2.DT_STRING,
    tf.bool: mace_pb2.DT_BOOL,
}


def tf_dtype_2_mace_dtype(tf_dtype):
    mace_dtype = TF_DTYPE_2_MACE_DTYPE_MAP.get(tf_dtype, None)
    if not mace_dtype:
        raise Exception("Not supported tensorflow dtype: " + tf_dtype)
    return mace_dtype
