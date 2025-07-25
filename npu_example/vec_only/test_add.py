lib_path = "./kernel_lib.so"

import torch
from typing import Callable, Optional, Literal, List, Union
from functools import partial

import argparse

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

a = torch.randn(M, N).half().npu()
b = torch.randn(M, N).half().npu()
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


bl = partial(torch.add, a, b, out=ref_c)

tl()
bl()
print(c, ref_c)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
