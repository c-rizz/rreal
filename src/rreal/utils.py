from typing import Callable
import torch as th
from rreal.nets.Parallel import Parallel
from torch.nn.utils.parametrizations import weight_norm
from adarl.utils.tensor_trees import TensorTree
import gymnasium as gym

def build_mlp_net(arch, input_size, output_size,  ensemble_size=1,
                    last_activation_class : Callable[[],th.nn.Module] = th.nn.Identity,
                    return_ensemble_mean = True,
                    hidden_activations : Callable[[],th.nn.Module] = th.nn.LeakyReLU,
                    return_ensemble_std : bool = False, use_torchscript : bool = False,
                    use_weightnorm : bool = False,
                    weight_init_multiplier = 1.0):
        
    if arch == "identity":
        if input_size != output_size:
            raise AttributeError(f"Requested identity mlp, but input_size!=output_size: {input_size} != {output_size}")
        net = Parallel([last_activation_class()], return_mean=return_ensemble_mean)
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
                layers.append(ll)
                if i < len(layersizes) - 2:
                    layers.append(hidden_activations())
            layers.append(last_activation_class())
            nets.append(th.nn.Sequential(*layers))
        net = Parallel(nets, return_mean=return_ensemble_mean, return_std=return_ensemble_std)
    else:
        raise AttributeError(f"Invalid arch {arch}")
    if weight_init_multiplier != 1:
        with th.no_grad():
            def scale_weights(m):
                if isinstance(m, th.nn.Linear):
                    m.weight *= weight_init_multiplier
                    m.bias *= weight_init_multiplier
            net.apply(scale_weights)
    if use_torchscript:
        net = th.compile(net)
    return net
