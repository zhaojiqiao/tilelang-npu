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
b = torch.empty(M, N).half().npu()

print("init successful!")
import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_


def tl():
    return lib.call(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
        stream)
tl()

ref_b = a.half()

torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)
