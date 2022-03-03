# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import torch
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy as default
from torch.distributed.fsdp.wrap import wrap_if_annotated
from torch.distributed.fsdp.wrap import wrap, enable_wrap

import torchdistx
from torchdistx import fake, deferred_init
from models import ShardedGPT, GPTSmallConfig, GPTLargeConfig, GPTXLConfig, GPTXXLConfig, GPTXXXLConfig, GPT13BConfig, GPT175BConfig

_VOCAB_SIZE = 3072
_BLOCK_SIZE = 128

def init_fn(module):
    deferred_init.materialize_module(module, recurse=False)

def _deferred_gpt(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    gpt = deferred_init.deferred_init(ShardedGPT, config=ct, device=torch.cuda.current_device())
    return gpt

def _deferred_gpt_wrap(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    with enable_wrap(wrapper_cls=FSDP, param_init_fns=init_fn):
        return wrap(deferred_init.deferred_init(ShardedGPT, config=ct, device=torch.cuda.current_device()))
#        return deferred_init.deferred_init(ShardedGPT(config=ct,device=torch.cuda.current_device()))
#        return wrap(deferred_init.deferred_init(ShardedGPT(config=ct, device=torch.cuda.current_device())))

def _regular_gpt_big(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    with enable_wrap(wrapper_cls=FSDP):
        return wrap(ShardedGPT(config=ct, device=torch.cuda.current_device()))

def _regular_gpt_big_wrap_everything(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    return FSDP(ShardedGPT(config=ct, device=torch.cuda.current_device()), fsdp_auto_wrap_policy=default)

def _regular_gpt(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    gpt = ShardedGPT(config=ct, device=torch.cuda.current_device())
    return gpt

class MyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.lin1 = nn.Linear(2, 2, bias=False, device=device)
        self.lin2 = nn.Linear(2, 2, bias=False, device=device)

    def forward(self, x):
        return self.lin2(self.lin1(x))


def _deferred_lambda():
    model = deferred_init.deferred_init(MyModel, "cuda")
    return model

def _regular_lambda():
    regular = MyModel(device="cuda")
    return regular

def run_test_with_deferred(deferred_lambda, regular_lambda):
    # Create model with deffered_init API

    init_deferred_start = torch.cuda.Event(enable_timing=True)
    init_deferred_end = torch.cuda.Event(enable_timing=True)
    init_regular_start = torch.cuda.Event(enable_timing=True)
    init_regular_end = torch.cuda.Event(enable_timing=True)

    # Test to make sure it is the same model parameters as regular FSDP
    # approach.
    def run_regular():
        init_regular_start.record()
        regular = regular_lambda()
        #regular.apply(init_fn)
        if isinstance(regular, FSDP):
            fsdp_regular = regular
        else:
            raise ValueError("Unexpected?")
    #        fsdp_regular = FSDP(regular, fsdp_auto_wrap_policy=wrap_if_annotated)
        init_regular_end.record()
        torch.cuda.synchronize()
        sec_conversion = 1000
        regular_time = init_regular_start.elapsed_time(init_regular_end) / sec_conversion
        total_fsdp_modules = 0
        for module in fsdp_regular.modules():
            if isinstance(module, FSDP):
                total_fsdp_modules+= 1
        if dist.get_rank() == 0: print(f"regular build time: {regular_time} sec {regular_time / total_fsdp_modules} per FSDP instance, {total_fsdp_modules} instances")

    run_regular()
    dist.barrier()
    torch.cuda.synchronize()
    def run_deferred():
        init_deferred_start.record()
        model = deferred_lambda()
        pol = default
        if not isinstance(model, FSDP):
            #fsdp_model = FSDP(model, param_init_fns=init_fn)
            fsdp_model = FSDP(model, fsdp_auto_wrap_policy=pol, param_init_fns=init_fn)
        else:
            fsdp_model = model
        init_deferred_end.record()
        sec_conversion = 1000
        deferred_time = init_deferred_start.elapsed_time(init_deferred_end) / sec_conversion
        total_fsdp_modules = 0
        for module in fsdp_model.modules():
            if isinstance(module, FSDP):
                total_fsdp_modules += 1
        if dist.get_rank() == 0: print(f"deferred build time: {deferred_time} sec {deferred_time / total_fsdp_modules} per fsdp instance, {total_fsdp_modules} instances")
        #print(fsdp_model)
        for p in fsdp_model.parameters():
            assert not fake.is_fake(p)
        print(" -- nothing is fake --")

    run_deferred()
    print("Initialized both FSDP models")
    return
    for m1, m2 in zip(fsdp_model.modules(), fsdp_regular.modules()):
        p1 = list(m1.parameters())
        p2 = list(m2.parameters())
        for x, y in zip(p1, p2):
            assert torch.allclose(x, y), f"{x} {y} "

    print(f"Initialized FSDP model, verified all params are equal!")

def run_test_with_meta():
    model = MyModel(device="meta")
    assert next(model.lin1.parameters()).is_meta
    assert next(model.lin2.parameters()).is_meta

    def init_fn(module):
        is_meta = any(t.is_meta for t in module.parameters())
        if is_meta: module.to_empty(device=torch.cuda.current_device())
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        try:
            with torch.no_grad():
                module.reset_parameters()
        except BaseException:
            pass

    # Eventually the API will be as simple as FSDP(
    #    model,
    #    fsdp_auto_wrap_policy=auto_wrap_policy
    # )
    # where we will use materialize_module() API to initialize the module
    fsdp_model = FSDP(
        model,
        fsdp_auto_wrap_policy=default,
        param_init_fns=init_fn,
    )
    print(fsdp_model)

    # Test to make sure it is the same model parameters as regular FSDP
    # approach.
    regular = MyModel(device="cuda")
    regular.apply(init_fn)
    fsdp_regular = FSDP(regular, fsdp_auto_wrap_policy=default)

    for m1, m2 in zip(fsdp_model.modules(), fsdp_regular.modules()):
        p1 = list(m1.parameters())
        p2 = list(m2.parameters())
        for x, y in zip(p1, p2):
            assert torch.allclose(x, y), f"{x} {y} "

    print(f"Initialized FSDP model")

def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=8)
    torch.cuda.set_device(rank)
#    d = _deferred_lambda
#    r = _regular_lambda

    gpt_config= GPTXXXLConfig
#    gpt_config = GPT13BConfig
#    gpt_config = GPT175BConfig
    d = partial(_deferred_gpt, cfg=gpt_config)
    r = partial(_regular_gpt_big_wrap_everything, cfg=gpt_config)
#    d = partial(_deferred_gpt_wrap, cfg=gpt_config)
#    r = partial(_regular_gpt_big, cfg=gpt_config)
    run_test_with_deferred(d, r)
#    run_test_with_meta()



if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=8, args=())
