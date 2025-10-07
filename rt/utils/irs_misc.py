# SPDX-FileCopyrightText: Copyright (c) 2025 Jean ACKER.
# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf

def log10(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log10` function.

    Simple extension to `tf.experimental.numpy.log10`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10>`__ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log10.html>`__ documentation.
    """
    return tf.cast(_log10(x), x.dtype)