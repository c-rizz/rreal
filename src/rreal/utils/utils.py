from __future__ import annotations
from typing import Callable
import torch as th
from rreal.nets.Parallel import Parallel
from torch.nn.utils.parametrizations import weight_norm
from adarl.utils.tensor_trees import TensorTree
import gymnasium as gym

def scale_layer_weights(m : th.nn.Module, multiplier, bias_offset : th.Tensor | float = 0.0):
    if isinstance(m, th.nn.Linear):
        m.weight *= multiplier
        m.bias *= multiplier
        m.bias += bias_offset
    elif len(list(m.parameters()))==0:
        pass
    else:
        raise RuntimeError(f"Unexpected module type {type(m)}")    
    
def build_mlp_net(arch, input_size, output_size,  ensemble_size=1,
                    last_activation_class : Callable[[],th.nn.Module] = th.nn.Identity,
                    return_ensemble_mean = True,
                    hidden_activations : Callable[[],th.nn.Module] = th.nn.LeakyReLU,
                    return_ensemble_std : bool = False, use_torchscript : bool = False,
                    use_weightnorm : bool = False,
                    weight_init_multiplier = 1.0,
                    layer_init_func : Callable[[th.nn.Module],None] | None = None,
                    last_layer_init_func : Callable[[th.nn.Module],None] | None = None,
                    use_jit_fork : bool = True) -> Parallel:
        
    if arch == "identity":
        if input_size != output_size:
            raise AttributeError(f"Requested identity mlp, but input_size!=output_size: {input_size} != {output_size}")
        net = Parallel([last_activation_class()], return_mean=return_ensemble_mean,
                       use_jit_fork=use_jit_fork)
    elif isinstance(arch, (list, tuple)):
        nets = []
        arch = [int(s) for s in arch]
        for _ in range(ensemble_size):
            layersizes = ([int(input_size)] + 
                            list(arch) + 
                            [int(output_size)])
            layers = []
            for i in range(len(layersizes)-1):
                ll = th.nn.Linear(layersizes[i],layersizes[i+1])
                if use_weightnorm:
                    ll = weight_norm(ll)
                with th.no_grad():
                    if (i < len(layersizes) - 2 or last_layer_init_func is None) and layer_init_func is not None:
                        layer_init_func(ll)
                    if i == len(layersizes) - 2 and last_layer_init_func is not None:
                        last_layer_init_func(ll)
                layers.append(ll)
                if i < len(layersizes) - 2:
                    layers.append(hidden_activations())
            layers.append(last_activation_class())
            nets.append(th.nn.Sequential(*layers))
        net = Parallel(nets, return_mean=return_ensemble_mean, return_std=return_ensemble_std,
                       use_jit_fork=use_jit_fork)
    else:
        raise AttributeError(f"Invalid arch {arch}")
    with th.no_grad():
        if weight_init_multiplier != 1:
            net.apply(lambda m: scale_layer_weights(m,weight_init_multiplier))
    if use_torchscript:
        net : Parallel = th.compile(net)
    return net

def split_params_for_weight_decay(model : th.nn.Module,
                                  weight_decay : float,
                                  decay_bias : bool = False,
                                  extra_kwargs : dict[str, th.Tensor|float] = {}) -> list[dict[str, th.Tensor|float]]:
    decay : list[th.Tensor] = []
    no_decay : list[th.Tensor] = []
    for name, param in model.named_parameters():
        if (name.endswith(".bias") and not decay_bias) or name.endswith(".weight_g") or name.endswith(".original1"):
            no_decay.append(param)
        else:
            decay.append(param)
    decay_group = {'params': decay,       'weight_decay': weight_decay}
    decay_group.update(extra_kwargs)
    no_decay_group = {'params': no_decay, 'weight_decay': 0.0}
    no_decay_group.update(extra_kwargs)
    return [decay_group, no_decay_group]

def get_params_with_decay_mask( model : th.nn.Module,
                                decay_bias : bool = False) -> tuple[list[th.Tensor], list[bool]]:
    should_decay : list[bool] = []
    params : list[th.Tensor] = []
    for name, param in model.named_parameters():
        should_decay.append((name.endswith(".bias") and not decay_bias) or name.endswith(".weight_g") or name.endswith(".original1"))
        params.append(param)
    return params, should_decay

def simplified_clip_grad_norm_(
    parameters: list[th.Tensor],
    max_norm: float,
    norm_type: float = 2.0
) -> th.Tensor:
    r"""Simplified version of torch's clip_grad_norm_.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    norms = th._foreach_norm(grads, norm_type)
    total_norm = th.linalg.vector_norm(th.stack(norms), norm_type)
    clip_coef_clamped = th.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    th._foreach_mul_(grads, clip_coef_clamped)
    return total_norm



def _use_grad_for_differentiable(func: Callable[_P, _T]) -> Callable[_P, _T]:
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

def torch_monkey_patch():
    import torch.optim.optimizer as optimizer
    optimizer._use_grad_for_differentiable = _use_grad_for_differentiable