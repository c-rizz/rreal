import sys
import functools
from typing import Callable, TypeVar, ParamSpec
import os


_T = TypeVar("_T")
_P = ParamSpec("_P")
def _use_grad_for_differentiable_graphbreakfix(func: Callable[_P, _T]) -> Callable[_P, _T]:
    def _use_grad(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        import torch._dynamo
        from typing import cast
        from torch.optim.optimizer import Optimizer
        self = cast(Optimizer, args[0])  # assume first positional arg is `self`
        prev_grad = torch.is_grad_enabled()
        required_grad = self.defaults["differentiable"]
        if required_grad == prev_grad:
            return func(*args, **kwargs)
        
        try:
            # Note on graph break below:
            # we need to graph break to ensure that aot respects the no_grad annotation.
            # This is important for perf because without this, functionalization will generate an epilogue
            # which updates the mutated parameters of the optimizer which is *not* visible to inductor, as a result,
            # inductor will allocate for every parameter in the model, which is horrible.
            # With this, aot correctly sees that this is an inference graph, and functionalization will generate
            # an epilogue which is appended to the graph, which *is* visible to inductor, as a result, inductor sees that
            # step is in place and is able to avoid the extra allocation.
            # In the future, we will either 1) continue to graph break on backward, so this graph break does not matter
            # or 2) have a fully fused forward and backward graph, which will have no_grad by default, and we can remove this
            # graph break to allow the fully fused fwd-bwd-optimizer graph to be compiled.
            # see https://github.com/pytorch/pytorch/issues/104053
            torch.set_grad_enabled(required_grad)
            torch._dynamo.graph_break()
            ret = func(*args, **kwargs)
        finally:
            torch._dynamo.graph_break()
            torch.set_grad_enabled(prev_grad)
        return ret

    functools.update_wrapper(_use_grad, func)
    return _use_grad

def torch_monkey_patch(): # this does not work
    # was_already_imported = "torch" in sys.modules
    # if was_already_imported:
    #     print(f"WARNING: torch_monkey_patch: torch was already imported before calling torch_monkey_patch(). {os.getpid()}")
    # import torch.optim.optimizer as optimizer
    # optimizer._use_grad_for_differentiable = _use_grad_for_differentiable
    # if was_already_imported:
    #     sys.modules.pop("torch", None)
    #     import torch as th
    #     import importlib
    #     importlib.reload(th)
    patch_masked()

def patch_masked():
    import torch
    import torch.masked
    from torch.types import _dtype as DType

    def nosync_masked_reduction_identity(op_name: str, input: torch.Tensor, *args):
        """Return identity value as scalar tensor of a reduction operation on
        given input, or None, if the identity value cannot be uniquely
        defined for the given input.

        The identity value of the operation is defined as the initial
        value to reduction operation that has a property ``op(op_identity,
        value) == value`` for any value in the domain of the operation.
        Or put it another way, including or excluding the identity value in
        a list of operands will not change the reduction result.

        See https://github.com/pytorch/rfcs/pull/27 for more information.

        """
        dtype: DType = input.dtype
        device = input.device
        op_name = op_name.rsplit(".", 1)[-1]  # lstrip module name when present
        if op_name in {"sum", "cumsum"}:
            return torch.tensor(0, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name in {"prod", "cumprod"}:
            return torch.tensor(1, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name in {"amax", "argmax", "logaddexp"}:
            if torch.is_floating_point(input):
                return torch.tensor(-torch.inf, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
            elif torch.is_signed(input) or dtype == torch.uint8:
                return torch.tensor(torch.iinfo(dtype).min, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name == "logsumexp":
            if torch.is_floating_point(input):
                return torch.tensor(-torch.inf, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
            elif torch.is_complex(input):
                return torch.tensor(-torch.inf + 0j, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
            elif torch.is_signed(input) or dtype == torch.uint8:
                return torch.tensor(torch.iinfo(dtype).min, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name in {"amin", "argmin"}:
            if torch.is_floating_point(input):
                return torch.tensor(torch.inf, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
            elif torch.is_signed(input) or dtype == torch.uint8:
                return torch.tensor(torch.iinfo(dtype).max, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name == "mean":
            # Strictly speaking, the identity value of the mean operation
            # is the mean of the input. Since the mean value depends on
            # the dim argument and it may be a non-scalar tensor, we
            # consider the identity value of the mean operation ambiguous.
            # Moreover, the mean value of empty input is undefined.
            return None
        elif op_name == "norm":
            ord = args[0] if args else 2
            if ord == float("-inf"):
                assert torch.is_floating_point(input), input.dtype
                return torch.tensor(torch.inf, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
            return torch.tensor(0, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name == "median":
            # We use NaN for now because the implementation is currently using torch.nanmedian
            # and NaN is the identity for that function since it gets ignored
            dtype = input.dtype if torch.is_floating_point(input) else torch.float
            return torch.tensor(torch.nan, dtype=dtype).to(device=device, non_blocking=device.type=="cuda")
        elif op_name in {"var", "std"}:
            return None
        raise NotImplementedError(f"identity of {op_name} on {dtype} input")
    torch.masked._ops._reduction_identity = nosync_masked_reduction_identity
