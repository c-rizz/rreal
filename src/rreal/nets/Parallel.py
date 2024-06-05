from typing import Final
import torch as th
import torch.nn as nn
from typing import List

# @th.jit.script
def _run_efficient(x : th.Tensor, modules : List[th.nn.Module], streams):
    current_stream = th.cuda.current_stream()
    outs = [None] * len(modules)
    inputs_num = x.size()[0]
    if inputs_num == 1:
        for i in range(len(modules)):
            with th.cuda.stream(streams[i]):
                streams[i].wait_stream(current_stream)
                outs[i]=modules[i](x[0])
    else:
        for i in range(len(modules)): # apparently should use streams or something like that
            with th.cuda.stream(streams[i]):
                streams[i].wait_stream(current_stream)
                outs[i]=modules[i](x[i])
    for s in streams:
        current_stream.wait_stream(s)
    return th.stack( outs, dim=1 )

# @th.jit.script
def _run_with_fork(x : th.Tensor, modules : List[th.nn.Module]):
    futures = [th.jit.fork(model, x) for model in modules]
    results = [th.jit.wait(fut) for fut in futures]
    return th.stack(results, dim=1)

def _run_simple(x : th.Tensor, modules : List[th.nn.Module]):
    outs = [None] * len(modules)
    inputs_num = x.size()[0]
    for i in range(len(modules)):
        outs[i]=modules[i](x[i])
    return th.stack( outs, dim=1 )

class Parallel(nn.ModuleList):
    """
    Parallelly runs provided modules. Returns one batch containig an ensemble of outputs in each element.
    E.g.: You have 3 submodules, each returning an output of size (5,), you input a (1024,10) batch, you get a (1024,3,5) output
    """
    def __init__(self, modules, return_mean : bool = False, return_std = False):
        super().__init__( modules )
        self._output_mean : Final[bool] = return_mean
        self._output_std : Final[bool] = return_std
        # self._streams = [th.cuda.Stream() for _ in range(len(modules))]

    def forward(self, x):

        # if isinstance(x,list) or isinstance(x,tuple):
        #     x_tens = th.stack(x, dim = 0)
        # else:
        #     x_tens = x.unsqueeze(0)

        # modules = list(self)
        

        # stacked_outs = _run_simple(x_tens, modules)

        # # streams = [th.cuda.Stream() for _ in range(len(modules))]
        # streams = [th.cuda.current_stream(device=th.cuda.current_device()) for _ in range(len(modules))]
        # # streams = self._streams
        # stacked_outs = _run_efficient(x_tens, modules, streams)

        # This gets parallelized if the moduled is compiled in torchscript
        # to compile you can do 'net = th.jit.script(Parallel(...))'
        futures = [th.jit.fork(model, x) for model in self]
        results = [th.jit.wait(fut) for fut in futures]
        stacked_outs = th.stack(results, dim=1)

        # modules = list(self)
        # results = [None]*len(modules)
        # for i in range(len(modules)):
        #     results[i] = modules[i](x)
        # stacked_outs = th.stack(results, dim=1)

        if self._output_mean:
            if self._output_std:
                return th.stack([th.mean(stacked_outs, dim=1), th.std(stacked_outs, dim=1)], dim=0)
            else:
                return th.mean(stacked_outs, dim=1)
        else:
            return stacked_outs

class MeanStd(nn.Module):
    def __init__(self, dim = 1):
        super().__init__()
        self._dim : Final[int] = dim
    def forward(self, x):
        return th.mean(x, dim=self._dim), th.std(x, dim=self._dim)