# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

N = 1024

def vec_add(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            T.copy(A[cid * block_N], A_VEC, [tail_size])
            T.copy(B[cid * block_N], B_VEC, [tail_size])

            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[cid * block_N], [tail_size])
    return main

def test_vec_add():
    func = vec_add(N, 1024)
    kernel = tilelang.engine.lower(func)
    print(kernel)

    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    file_name = './output/' + curr_name
    with open(file_name, 'w') as f:
        f.write(kernel)

if __name__ == "__main__":
    test_vec_add()
