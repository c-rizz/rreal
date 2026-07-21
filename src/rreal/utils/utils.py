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
    elif isinstance(m, th.nn.utils.parametrize.ParametrizationList):
        pass # its original0/original1 were already scaled through the parametrized .weight above
    elif len(list(m.parameters(recurse=False)))==0:
        pass # container, or parameter-free leaf
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


def copy_net_state(src : th.nn.Module, dst : th.nn.Module, strict = True):
    if strict:
        param_mapping = {}
        buff_mapping = {}
        for name, src_param in src.named_parameters():
            dst_param = dst.get_parameter(name)
            param_mapping[name] = (src_param, dst_param)
        for name, src_buffer in src.named_buffers():
            dst_buffer = dst.get_buffer(name)
            buff_mapping[name] = (src_buffer, dst_buffer)
        for name, dst_param in dst.named_parameters():
            if name not in param_mapping:
                raise KeyError(f"Key {name} found in destination state dict but not in source state dict")
        for name, dst_buffer in dst.named_buffers():
            if name not in buff_mapping:
                raise KeyError(f"Key {name} found in destination state dict but not in source state dict")
        for _,(src_tensor, dst_tensor) in param_mapping.items():
            dst_tensor.data.copy_(src_tensor.data)
        for _,(src_tensor, dst_tensor) in buff_mapping.items():
            dst_tensor.data.copy_(src_tensor.data)
    else:
        for name, src_param in src.named_parameters():
            dst.get_parameter(name).data.copy_(src_param.data)
        for name, src_buffer in src.named_buffers():
            dst.get_buffer(name).data.copy_(src_buffer.data)

def update_net_state(src : th.nn.Module, dst : th.nn.Module, strict = True, tau = 0.005):
    if strict:
        param_mapping = {}
        buff_mapping = {}
        for name, src_param in src.named_parameters():
            dst_param = dst.get_parameter(name)
            param_mapping[name] = (src_param, dst_param)
        for name, src_buffer in src.named_buffers():
            dst_buffer = dst.get_buffer(name)
            buff_mapping[name] = (src_buffer, dst_buffer)
        for name, dst_param in dst.named_parameters():
            if name not in param_mapping:
                raise KeyError(f"Key {name} found in destination state dict but not in source state dict")
        for name, dst_buffer in dst.named_buffers():
            if name not in buff_mapping:
                raise KeyError(f"Key {name} found in destination state dict but not in source state dict")
        for _,(src_tensor, dst_tensor) in param_mapping.items():
            dst_tensor.data.copy_(tau * src_tensor.data + (1-tau) * dst_tensor.data)
        for _,(src_tensor, dst_tensor) in buff_mapping.items():
            dst_tensor.data.copy_(tau * src_tensor.data + (1-tau) * dst_tensor.data)
    else:
        for name, src_param in src.named_parameters():
            dst.get_parameter(name).data.copy_(src_param.data * tau + dst.get_parameter(name).data * (1-tau))
        for name, src_buffer in src.named_buffers():
            dst.get_buffer(name).data.copy_(src_buffer.data * tau + dst.get_buffer(name).data * (1-tau))