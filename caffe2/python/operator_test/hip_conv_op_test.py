from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class HIPConvTest(hu.HypothesisTestCase):
    @given(stride=st.integers(2, 2),
           pad=st.integers(3, 3),
           kernel=st.integers(7, 7),
           size=st.integers(224, 224),
           input_channels=st.integers(3, 3),
           output_channels=st.integers(64, 64),
           batch_size=st.integers(1, 1),
           **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_hip_convolution(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, gc, dc):
        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            engine="MIOPEN",
        )
        X = np.random.rand(
            batch_size, input_channels, size, size
        ).astype(np.float32) + 0.5
        w = np.random.rand(
                output_channels, input_channels, kernel, kernel
        ).astype(np.float32) + 0.5
        b = np.random.rand(output_channels).astype(np.float32) + 0.5

        inputs = [X, w, b]
        self.assertDeviceChecks(dc, op, inputs, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
