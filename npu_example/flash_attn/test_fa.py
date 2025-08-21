lib_path = "./kernel_lib.so"

import torch
from typing import Callable, Optional, Literal, List, Union


SEQ_LEN = 16384

DIM = 128

Q = torch.randn(SEQ_LEN, DIM).half().npu()
K = torch.randn(SEQ_LEN, DIM).half().npu()
V = torch.randn(SEQ_LEN, DIM).half().npu()

Output = torch.empty(SEQ_LEN, DIM).float().npu()

print("init successful!")
import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_

def tl():
    return lib.call(
        # noqa: E501
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_void_p(Output.data_ptr()),
        stream)




tl()

print(Output)

