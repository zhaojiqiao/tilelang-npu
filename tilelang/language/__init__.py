# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Optional
# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from .tir import (
    prim_func,  # noqa: F401
)
from .tir.ir import *  # noqa: F401
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import (
    ptr,  # noqa: F401
    make_tensor,  # noqa: F401
    Buffer,  # noqa: F401
    Tensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .warpgroup import ws, rs, set_flag, wait_flag, Scope  # noqa: F401
from .allocate import (
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_var,  # noqa: F401
    alloc_L0A,  # noqa: F401
    alloc_L0B,  # noqa: F401
    alloc_L0C,  # noqa: F401
    alloc_L1,  # noqa: F401
    alloc_ub,  # noqa: F401
)
from .copy import copy, c2d_im2col, npu_copy_v2 as copy  # noqa: F401, F811
from .gemm import GemmWarpPolicy, gemm  # noqa: F401
from .fill import fill, clear  # noqa: F401
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    cumsum,  # noqa: F401
)
from .print import print  # noqa: F401
from .customize import (
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    atomic_addx4,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
    npu_gemm as gemm,  # noqa: F401, F811
    # npu_copy as copy,  # noqa: F401, F811
    npu_add as tile_add,  # noqa: F401, F811
)
from .customize_npuir import  (
    npuir_copy as copy,
    npuir_add,
    npuir_sub,
    npuir_max,
    npuir_min,
    npuir_mul,
    npuir_div,
    npuir_exp,
    npuir_dot,
    npuir_load_nd2nz,
    npuir_store_fixpipe,
    npuir_brc,
    npuir_cast,
    npuir_reduce,
    rs,
    set_flag,
    wait_flag,
    pipe_barrier,
    block_barrier,
    subblock_barrier,
    sync_block_set,
    sync_block_wait,
    Scope
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import *  # noqa: F401

from .memscope import *  # noqa: F401


def symbolic(name: str, dtype: str = "int32"):
    return tir.Var(name, dtype)


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    return attr(None, "threadblock_swizzle_pattern",
                f"tl::{device_func}<{panel_size}>") if enable else None


def annotate_layout(layout_map: Dict):
    """Annotate the layout of the buffer

    Args:
        layout_map (Dict): a dictionary of buffer to layout

    Returns:
        block_attr: a block attribute
    
    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_layout({A_shared: layout})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # layout_map is a dictionary of buffer to layout
    layout_map = {buffer.data: layout for buffer, layout in layout_map.items()}
    return block_attr({"layout_map": layout_map})


def annotate_padding(padding_map: Dict):
    """Annotate the padding of the buffer

    Args:
        padding_map (dict): a dictionary of buffer to padding value

    Returns:
        block_attr: a block attribute
    
    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_padding({A_shared: pad_value})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i - 10, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # padding_map is a dictionary of buffer to padding value
    _padding_map = {}
    for buffer, padding_value in padding_map.items():
        # assert not global
        assert buffer.scope() != "global", "padding can only be applied to global buffers"
        _padding_map[buffer.data] = padding_value
    return block_attr({"padding_map": _padding_map})


def import_source(source: Optional[str] = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None


def init_flag(fmap):
    inst = ""
    for src, d in fmap.items():
        for dst, stages in d.items():
            for stage in stages:
                inst += f"AscendC::SetFlag<AscendC::HardEvent::{src}_{dst}>({stage});\n"

    return attr(None, "init_flag", inst)


def clear_flag(fmap):
    inst = ""
    for src, d in fmap.items():
        for dst, stages in d.items():
            for stage in stages:
                inst += f"AscendC::WaitFlag<AscendC::HardEvent::{src}_{dst}>({stage});\n"

    @macro
    def _get_inst():
        with attr(None, "clear_flag", inst):
            call_extern("handle", "...")

    # return attr(call_extern("handle", "..."), "clear_flag", inst)
    return _get_inst()


def npu_use_swizzle(m, n, k, block_m, block_n, off=1, dir=0):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    # device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    return attr(
        None, "threadblock_swizzle_pattern",
        f"tl::ascend::thread_block_swizzle<{m}, {n}, {k}, {block_m}, {block_n}, {off}, {dir}>")


del use_swizzle


# let npu_use_swizzle is aliased to use_swizzle
def use_swizzle(m, n, k, block_m, block_n, off=1, dir=0):
    """Alias for npu_use_swizzle with proper signature for function hints."""
    return npu_use_swizzle(m, n, k, block_m, block_n, off, dir)
