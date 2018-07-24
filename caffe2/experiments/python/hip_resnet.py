from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx

import argparse
import os
import shutil

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python.onnx.helper import c2_native_run_net

import numpy as np

steps = 79

def parse_args():
    parser = argparse.ArgumentParser('hip_resnet')
    return parser.parse_args()


def assert_close(o1, o2):
    if not len(o1) == len(o2):
        raise ValueError(
            'Different number of outputs: {} vs. {}'.format(len(o1), len(o2)))

    for v1, v2 in enumerate(zip(o1, o2)):
        print(len(o1))
        for v1, v2 in zip(o1, o2):
            if not v1.shape == v2.shape:
                raise ValueError(
                    'Different shape: {} vs. {}'.format(v1.shape, v2.shape))
        print(v1.shape)
        print(np.max(np.abs(v1 - v2)))
        np.testing.assert_almost_equal(v1, v2, decimal=5)
        # np.testing.assert_allclose(
        #     v1,
        #     v2,
        #     rtol=0.01,
        #     atol=0.01,)


def main():
    args = parse_args()

    init = caffe2_pb2.NetDef()
    with open('/tmp/resnet50/init_net.pb', 'rb') as f:
        init.ParseFromString(f.read())
    predict = caffe2_pb2.NetDef()
    with open('/tmp/resnet50/predict_net.pb', 'rb') as f:
        predict.ParseFromString(f.read())

    tmp_dir = "/tmp/tensors"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    outputs = {}
    inputs = [np.random.randn(1, 3, 224, 224).astype(dtype=np.float32) + 2]
    for dc_s in ['HIP', 'CPU']:
        dc = core.DeviceOption(getattr(caffe2_pb2, dc_s))
        for op in init.op:
            op.device_option.CopyFrom(dc)
        init.device_option.CopyFrom(dc)

        for op in predict.op:
            op.device_option.CopyFrom(dc)
        predict.device_option.CopyFrom(dc)
        ws, results = c2_native_run_net(init, predict, inputs, steps=steps)
        outputs[dc_s] = results
        for t_s in predict.op[steps - 1].input:
            t_s_f = t_s.replace('/', '_')
            name = '_'.join([dc_s, t_s_f])
            with open(os.path.join(tmp_dir, name), 'wb') as f:
                np.save(f, ws.FetchBlob(t_s))

    assert_close(outputs['CPU'], outputs['HIP'])


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    np.random.seed(0)
    main()
