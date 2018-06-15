#include "caffe2/operators/hip/operator_fallback_hip.h"
#include "caffe2/operators/given_tensor_fill_op.h"
#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/operators/dropout_op.h"


namespace caffe2 {
    REGISTER_HIP_OPERATOR(
        GivenTensorFill,
        HIPFallbackOp<GivenTensorFillOp<float, CPUContext>>);

    REGISTER_HIP_OPERATOR(
        FC,
        HIPFallbackOp<FullyConnectedOp<CPUContext>>);

    REGISTER_HIP_OPERATOR(
        Sum,
        HIPFallbackOp<SumOp<CPUContext>>);

    REGISTER_HIP_OPERATOR(
        Dropout,
        HIPFallbackOp<DropoutOp<float, CPUContext>>);
}
