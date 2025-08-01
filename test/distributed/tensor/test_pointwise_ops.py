# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from collections.abc import Sequence
from typing import Any, Callable, Optional
from unittest import skip

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorOpTestBase,
    skip_unless_torch_gpu,
)


def no_op():
    return None


def deepcopy_convert_to_dtensor(
    val: Any,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Any:
    """
    Recursively convert (over Sequence and Dict types) Tensors into DTensors.

    :param device_mesh: the DeviceMesh to use.
    :param placements: the Placement list to use.
    :return: the transformed structure.
    """

    def f(x):
        if isinstance(x, Tensor) and not isinstance(x, DTensor):
            return distribute_tensor(
                x,
                device_mesh=device_mesh,
                placements=placements,
            )
        return x

    return pytree.tree_map(f, [val])[0]


def deepcopy_convert_from_dtensor(val: Any) -> Any:
    """
    Recursive convert any DTensor to local Tensor.

    :param val: the structure to coerce.
    :return: the coerced structure.
    """

    def f(x):
        if isinstance(x, DTensor):
            return x.full_tensor()
        return x

    return pytree.tree_map(f, [val])[0]


class DistElementwiseOpsTest(DTensorOpTestBase):
    def _compare_pairwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        op: Callable,
        pre_op_fn: Optional[Callable] = None,
        args: Sequence[Any] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        if not kwargs:
            kwargs = {}

        dargs = deepcopy_convert_to_dtensor(
            args,
            device_mesh=device_mesh,
            placements=placements,
        )
        dkwargs = deepcopy_convert_to_dtensor(
            kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )

        pre_op_fn()

        # run the reference first, in case the call is broken;
        # it's better to debug an incorrect call at this point.
        reference_result = op(*args, **kwargs)

        pre_op_fn()

        dist_result = op(*dargs, **dkwargs)

        collected_result = deepcopy_convert_from_dtensor(dist_result)

        self.assertEqualOnRank(reference_result, collected_result)

    # TODO: We need to add CPU tests for ops in the future.
    def _run_sharded_elementwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        pre_op_fn: Optional[Callable] = None,
        input_size: Sequence[int],
        op: Callable,
        **kwargs,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        input_tensor = torch.randn(
            *input_size,
            device=self.device_type,
            requires_grad=True,
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            pre_op_fn=pre_op_fn,
            op=op,
            args=(input_tensor,),
            kwargs=kwargs,
        )

    def test_partial_add(self):
        device_mesh = self.build_device_mesh()
        d_1 = DTensor.from_local(torch.rand(2, 2), device_mesh, [Partial()])
        d_2 = DTensor.from_local(torch.rand(2, 2), device_mesh, [Partial()])
        d_3 = d_1 + d_2
        self.assertTrue(d_3._spec.placements[0].is_partial())

    def test_activations(self):
        device_mesh = self.build_device_mesh()
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 12),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.sigmoid,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.sigmoid,
        )

    @skip(
        "testing RNG based ops is broken: https://github.com/pytorch/PiPPy/issues/494"
    )
    def test_dropout(self):
        device_mesh = self.build_device_mesh()

        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.4,
            training=False,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.5,
            training=True,
        )

    @skip_unless_torch_gpu
    def test_dropout_backward(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]

        input_size = (8, 5)

        grad_output = torch.rand(
            input_size,
            device=self.device_type,
            requires_grad=True,
        )
        mask = (
            torch.rand(
                input_size,
                device=self.device_type,
                requires_grad=False,
            )
            < 0.8
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            op=torch.ops.aten.native_dropout_backward,
            kwargs=dict(
                grad_output=grad_output,
                mask=mask,
                scale=0.3,
            ),
        )

    def test_dropout_errors(self):
        device_mesh = self.build_device_mesh()
        with self.assertRaisesRegex(RuntimeError, "supported"):
            self._run_sharded_elementwise_ops(
                device_mesh=device_mesh,
                placements=[Partial("sum")],
                input_size=(8, 5),
                op=torch.nn.functional.dropout,
            )

    def test_mul_out(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(self.rank)
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        other_tensor = torch.randn(*input_size, device=self.device_type)
        other_dtensor = DTensor.from_local(other_tensor, device_mesh, shard_spec)

        output_tensor = torch.randn(*input_size, device=self.device_type)
        output_dtensor = DTensor.from_local(output_tensor, device_mesh, shard_spec)
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())

    def test_mul_partial(self):
        # we only test the partial behavior for mul op as other placement
        # behaviors should be well tested in test_dtensor_ops.py
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        # 1. simple test for partial * partial
        d_1 = DTensor.from_local(torch.ones(2, 2), device_mesh, [Partial()])
        d_2 = DTensor.from_local(torch.ones(2, 2), device_mesh, [Partial()])
        with comm_mode:
            d_3 = d_1 * d_2
        comm_counts = comm_mode.get_total_counts()
        self.assertEqual(comm_counts, 1)
        self.assertTrue(isinstance(d_3, DTensor))
        self.assertEqual(d_3.placements, (Partial(),))
        self.assertEqual(d_3.to_local(), torch.ones(2, 2) * (self.world_size))

        # 2. test the partial input DTensor * scalar/replicate input
        input = torch.full((8, 8), 1.0, device=self.device_type)

        # test for different types of other inputs
        other_inps = (
            2.0,  # scalar
            torch.tensor(2.0, device=self.device_type),  # scalar tensor
            torch.full((8, 8), 2.0, device=self.device_type),  # tensor
        )

        for partial_op in ["sum", "avg"]:
            expected_p_out = (
                input * self.world_size * 2.0 if partial_op == "sum" else input * 2.0
            )

            d_input = DTensor.from_local(input, device_mesh, [Partial(partial_op)])

            for other_inp in other_inps:
                if isinstance(other_inp, Tensor) and other_inp.numel() > 1:
                    d_other = distribute_tensor(other_inp, device_mesh, [Replicate()])
                else:
                    d_other = other_inp

                with comm_mode:
                    z = d_input * d_other

                comm_counts = comm_mode.get_total_counts()
                self.assertEqual(comm_counts, 0)
                self.assertTrue(isinstance(z, DTensor))
                self.assertEqual(z.placements, (Partial(partial_op),))
                self.assertEqual(z.full_tensor(), expected_p_out)

        # test other partial to assert the partial not getting propagated
        d_input = DTensor.from_local(input, device_mesh, [Partial("max")])
        d_other = distribute_tensor(torch.ones(8, 8), device_mesh, [Replicate()])

        z = d_input * d_other
        self.assertEqual(z.placements, (Replicate(),))
        self.assertEqual(z.to_local(), input)


if __name__ == "__main__":
    run_tests()
