lib_path = "./kernel_lib.so"

import torch
from typing import Callable, Optional, Literal, List, Union
from functools import partial

import argparse

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=16384, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=16384, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=16384, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    fast_flush: bool = True,
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> Union[float, List[float]]:
    """Benchmarks the runtime of a PyTorch function.
    
    This function handles:
    - L2 cache flushing between runs for consistent timing
    - Automatic warmup and repeat count calculation
    - Optional gradient clearing for backward passes
    - Multiple measurement modes (mean, median, min, max)
    
    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds
        rep: Target number of repetitions
        _n_warmup: Override for number of warmup iterations
        _n_repeat: Override for number of timing iterations
        grad_to_none: Tensors whose gradients should be cleared between runs
        quantiles: Optional performance percentiles to compute
        fast_flush: Whether to use faster L2 cache flushing
        return_mode: How to aggregate timing results ("mean", "median", "min", "max")
        
    Returns:
        float: Aggregated runtime in milliseconds
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.npu.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    # if fast_flush:
    #     cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    # else:
    cache = torch.empty(int(202e6), dtype=torch.int8, device="npu")

    # Estimate the runtime of the function
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.npu.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat
    start_event = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.npu.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.npu.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
ref_c = torch.empty(M, N).half().npu()
print("init successful!")
import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_


def tl():
    return lib.call(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()),
        stream)


bl = partial(torch.mm, a, b, out=ref_c)

tl()
bl()

print(c, ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

tl_time = do_bench(tl)
bl_time = do_bench(bl)

print(tl_time, bl_time)

FLOP = 2 * M * N * K

print(f"TL TFlops: { FLOP / tl_time * 1e-9:.3f}")
print(f"Baseline TFlops: { FLOP / bl_time * 1e-9:.3f}")
