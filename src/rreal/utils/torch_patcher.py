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
    was_already_imported = "torch" in sys.modules
    if was_already_imported:
        print(f"WARNING: torch_monkey_patch: torch was already imported before calling torch_monkey_patch(). {os.getpid()}")
    import torch.optim.optimizer as optimizer
    optimizer._use_grad_for_differentiable = _use_grad_for_differentiable
    if was_already_imported:
        sys.modules.pop("torch", None)
        import torch as th
        import importlib
        importlib.reload(th)