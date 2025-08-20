lib_path = "./kernel_lib.so"

import torch
from functools import partial

import argparse

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=128, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

a = torch.randn(M, N).float().npu()
b = torch.empty(M, 1).float().npu()

print("init successful!")
import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_


def tl():
    return lib.call(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
        stream)
tl()

ref_b = torch.sum(a, dim=1, keepdim=True)

print(b)
print(ref_b)

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
