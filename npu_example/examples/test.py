lib_path = "./kernel_lib.so"

import torch

import argparse

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=16384, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=16384, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=16384, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
print("init successful!")
import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_


def tl():
    return lib.call(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()),
        stream)


tl()

ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
