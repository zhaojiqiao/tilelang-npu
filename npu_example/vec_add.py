# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


def vec_add(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            with T.rs("MTE2"):
                T.copy(A[bx * block_M, by * block_N], A_VEC)
                T.copy(B[bx * block_M, by * block_N], B_VEC)
                T.set_flag("V", 0)

            with T.rs("V"):
                T.wait_flag("MTE2", 0)
                T.tile_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("MTE3", 0)

            with T.rs("MTE3"):
                T.wait_flag("V", 0)
                T.copy(C_VEC, C[bx * block_M, by * block_N])

    return main


func = vec_add(M, N, K, 128, 256)

kernel = tilelang.engine.lower(func)
print(kernel)