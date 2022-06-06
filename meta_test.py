# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import torch
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
FullyShardedDataParallel = FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as default
#from torch.distributed.fsdp.wrap import wrap_if_annotated
from torch.distributed.fsdp.wrap import wrap, enable_wrap

import torchdistx
from torchdistx import fake, deferred_init
from models import ShardedGPT, GPTSmallConfig, GPTLargeConfig, GPTXLConfig, GPTXXLConfig, GPTXXXLConfig, GPT13BConfig, GPT175BConfig

def sync_all_device():

    # setup() has already configured CUDA_VISIBLE_DEVICES such that each
    # process exclusively works on its own set of devices. So it's safe to
    # do device sync here
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)

_VOCAB_SIZE = 3072
_BLOCK_SIZE = 128
_SEC_CONVERSION = 1000
# inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device="cuda:0")
def init_fn(module):
    assert not isinstance(module, FSDP)
    check_fn = lambda k: not isinstance(k, FSDP)
    deferred_init.materialize_module(module, check_fn=check_fn)

def _deferred_gpt(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    gpt = deferred_init.deferred_init(ShardedGPT, config=ct, device=torch.cuda.current_device())
    end_ev.record()
    torch.cuda.synchronize()
    difftime = start_ev.elapsed_time(end_ev) / 1000
    if dist.get_rank() == 0: print(f"just deferred init took {difftime} ")
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    gpt = FSDP(gpt, auto_wrap_policy=default)
    end_ev.record()
    torch.cuda.synchronize()
    difftime = start_ev.elapsed_time(end_ev) / 1000
    if dist.get_rank() == 0: print(f"FSDP wrap after deferred init took {difftime} ")
    return gpt

# def _deferred_gpt_wrap(cfg):
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#     ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
#     with enable_wrap(wrapper_cls=FSDP):
#         return wrap(deferred_init.deferred_init(ShardedGPT, config=ct, device=torch.cuda.current_device()))
#        return deferred_init.deferred_init(ShardedGPT(config=ct,device=torch.cuda.current_device()))
#        return wrap(deferred_init.deferred_init(ShardedGPT(config=ct, device=torch.cuda.current_device())))

def _regular_gpt_big(cfg):
    import time
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
    from datetime import timedelta
    pg = dist.new_group(backend="nccl", timeout=timedelta(seconds=75))
#    print(" -- creating model --")
    s = time.time()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    model = ShardedGPT(config=ct, device=torch.cuda.current_device())
    # model.cpu()
    end_ev.record()
    torch.cuda.synchronize()
    difftime = start_ev.elapsed_time(end_ev) / 1000
    if dist.get_rank() == 0:
        print(f" -- created plain model in {difftime}, wrapping now --")
    torch.cuda.synchronize()
    se, ee = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    se.record()
    model = FSDP(model, auto_wrap_policy=default, device_id=torch.cuda.current_device())
    ee.record()
    torch.cuda.synchronize()
    difftime = se.elapsed_time(ee) / 1000
    taken =difftime
    #model = FSDP(ShardedGPT(config=ct, device=torch.device("cpu")), auto_wrap_policy=default)
    if dist.get_rank() == 0:
        print(f"Created fsdp model wrapping everything. Took {taken}")
    return model
    exit(0)
#    model = ShardedGPT(config=ct, device=torch.device("cpu"))
#    print("created model not FSDP")
    return model
    with enable_wrap(wrapper_cls=FSDP, process_group=pg):
    #    return wrap(ShardedGPT(config=ct, device=torch.cuda.current_device()))
        return wrap(ShardedGPT(config=ct, device=torch.cuda.current_device()))

# def _regular_gpt_big_wrap_everything(cfg):
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#     ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
#     return FSDP(ShardedGPT(config=ct, device=torch.cuda.current_device()), fsdp_auto_wrap_policy=default)

# def _regular_gpt(cfg):
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#     ct = cfg(vocab_size=_VOCAB_SIZE, block_size=_BLOCK_SIZE)
#     gpt = ShardedGPT(config=ct, device=torch.cuda.current_device())
#     return gpt

class MyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.lin1 = nn.Linear(2, 2, bias=False, device=device)
        self.lin2 = nn.Linear(2, 2, bias=False, device=device)

    def forward(self, x):
        return self.lin2(self.lin1(x))


# def _deferred_lambda():
#     model = deferred_init.deferred_init(MyModel, "cuda")
#     return model

# def _regular_lambda():
#     regular = MyModel(device="cuda")
#     return regular

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
#        regular.apply(init_fn)
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
        assert isinstance(fsdp_regular, FullyShardedDataParallel)
        return fsdp_regular

#    print(" -- starting regular --")
    if dist.get_rank() == 0: print("starting regualr", flush=True)
#    fsdp_regular = run_regular()
    dist.barrier()
    torch.cuda.synchronize()
    # return
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
        torch.cuda.synchronize()
        dist.barrier()
        deferred_time = init_deferred_start.elapsed_time(init_deferred_end) / sec_conversion
        total_fsdp_modules = 0
        for module in fsdp_model.modules():
            if isinstance(module, FSDP):
                total_fsdp_modules += 1
        if dist.get_rank() == 0: print(f"deferred build time: {deferred_time} sec {deferred_time / total_fsdp_modules} per fsdp instance, {total_fsdp_modules} instances")
        #print(fsdp_model)
        for p in fsdp_model.parameters():
            assert not fake.is_fake(p) and not fake.is_fake(p.data)
        assert isinstance(fsdp_model, FullyShardedDataParallel)
        if dist.get_rank() == 0:
            print(" --- META device is validated nothing is fake", flush=True)
        return fsdp_model
#        print(" -- nothing is fake --")

    import time
    start = time.time()
    dist.barrier()
    fsdp_model = run_deferred()
    torch.cuda.synchronize()
    dist.barrier()
    end = time.time()
    k = end - start
    print(f"Rank {dist.get_rank()} time: {k}", flush=True)
    if dist.get_rank() == 0:
        print("Initialized both FSDP models")
    # for m1, m2 in zip(fsdp_model.modules(), fsdp_regular.modules()):
    #     p1 = list(m1.parameters())
    #     p2 = list(m2.parameters())
    #     for x, y in zip(p1, p2):
    #         assert torch.allclose(x, y), f"{x} {y} "

    # print(f"Initialized FSDP model, verified all params are equal!", flush=True)
    dist.barrier()
    print(f"Rank {dist.get_rank()} starting forward pass", flush=True)
    # run deferred fwd + backward
    # torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device="cuda:0")
    inputs = torch.randint(0, _VOCAB_SIZE, (1, _BLOCK_SIZE), device=torch.cuda.current_device())
    out = fsdp_model(inputs).sum()
    out.backward()
#    sync_all_device()
    dist.barrier()
    torch.cuda.synchronize()
    print("Fwd + backward ran")
    return
    # print("ran fwd + backward")



def worker(rank):
    torch.cuda.set_device(rank)
    global_rank = os.environ["RANK"]
    global_ws = int(os.environ["WORLD_SIZE"]) * 8
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
#    os.environ["NCCL_DEBUG"] = "INFO"
    try:
        ifname = os.environ["NCCL_SOCKET_IFNAME"]
    except:
        ifname ="not set"
    print(
        f"init for local rank {rank} global {global_rank} ws={global_ws},{addr} {port}, {ifname}",
        flush=True
    )
    dist.init_process_group("nccl", rank=rank, world_size=global_ws)
    print(f"DOne init for local {rank}, global {global_rank}", flush=True)
    dist.barrier()
    # return
#    d = _deferred_lambda
#    r = _regular_lambda

    #gpt_config= GPTXXXLConfig
    #gpt_config = GPTXXXLConfig
#    gpt_config = GPTLargeConfig
    gpt_config = GPT13BConfig
    gpt_config=GPTSmallConfig
    gpt_config = GPTXXXLConfig
    gpt_config = GPT13BConfig
    gpt_config = GPTSmallConfig
    gpt_config = GPTXXXLConfig
#    gpt_config = GPT175BConfig
#    d = partial(_deferred_gpt, cfg=gpt_config)
    r = partial(_regular_gpt_big, cfg=gpt_config)
    d = partial(_deferred_gpt, cfg=gpt_config)
#    r = partial(_regular_gpt_big, cfg=gpt_config)
    run_test_with_deferred(d, r)
    exit(0)
#    run_test_with_meta()

def main():
    pass


if __name__ == "__main__":
    # os.environ["WORLD_SIZE"] = "2"
    print("script invoked")
    # exit(0)
    mp.spawn(worker, nprocs=8, args=())
