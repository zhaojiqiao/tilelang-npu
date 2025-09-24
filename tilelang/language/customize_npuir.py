"""The language interface for tl programs."""

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Var
from typing import List, Union, Optional
from tvm import ir, tir
from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm._ffi import register_object
from tilelang import _ffi_api
from .kernel import get_thread_bindings, get_thread_extents, FrameStack
import threading

from tilelang.language.copy import buffer_region_to_tile_region, buffer_load_to_tile_region, region


def _get_extent(data):
    if isinstance(data, tir.Var) and T.has_let_value(data):
        data = T.get_let_value(data)
    result = []
    if isinstance(data, tir.Buffer):
        result = data.shape
    elif isinstance(data, tir.BufferRegion):
        result = [x.extent for x in data.region]
    return result

def _buffer_to_tile_region_with_extent(buffer: tir.Buffer, access_type: str, extent:[]):
    """Convert a TVM buffer to a tile region descriptor.

    Args:
        buffer (tir.Buffer): The buffer to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        extent ([]): buffer extent

    Returns:
        tir.Call: A region descriptor covering the entire buffer
    """
    mins = [0 for _ in buffer.shape]
    return region(T.BufferLoad(buffer, mins), access_type, *extent)

def _to_region(data, access_type, extent):
    if isinstance(data, tir.Var) and T.has_let_value(data):
        data = T.get_let_value(data)
    if isinstance(data, tir.Buffer):
        return _buffer_to_tile_region_with_extent(data, access_type, extent)
    elif isinstance(data, tir.BufferRegion):
        return buffer_region_to_tile_region(data, access_type, extent[-len(data.buffer.shape):])
    elif isinstance(data, tir.IntImm) or isinstance(data, tir.FloatImm):
        return data
    else:
        return buffer_load_to_tile_region(data, access_type, extent[-len(data.buffer.shape):])


def npuir_copy(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    dst: Union[tir.Buffer, tir.BufferLoad],
    size: [] = []
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        size ([]): buffer extent

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    if size == []:
        src_extent = _get_extent(src)
        dst_extent = _get_extent(dst)
        assert src_extent or dst_extent, "Can't deduce copy extents from args"
        src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
        dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
        extent = max(src_extent, dst_extent)
    else:
        extent = size
    src = _to_region(src, "r", extent)
    dst = _to_region(dst, "w", extent)

    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_copy"), src, dst)


def npuir_add(A, B, C):
    """npuir add at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_add operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_add"), A, B, C)

def npuir_sub(A, B, C):
    """npuir sub at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_sub operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sub"), A, B, C)

def npuir_mul(A, B, C):
    """npuir mul at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_mul operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_mul"), A, B, C)

def npuir_div(A, B, C):
    """npuir div at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_div operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_div"), A, B, C)

def npuir_max(A, B, C):
    """npuir max at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_max operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_max"), A, B, C)

def npuir_min(A, B, C):
    """npuir min at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_min operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_min"), A, B, C)


def npuir_exp(A, B):
    """npuir exponent at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_exp operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "w", _get_extent(B))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_exp"), A, B)

def npuir_dot(A: Union[tir.Buffer, tir.Var],
    B: Union[tir.Buffer, tir.Var],
    C: Union[tir.Buffer, tir.Var],
    size: [] = [], initC: bool = False, a_transpose: bool = False, b_transpose: bool = False):
    """npuir dot at tile-level. C = C + A * B.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
        initC (bool): whether to initialize L0C value to zero (C = A * B)
        a_transpose (bool): Matrix A is transposed before load
        b_transpose (bool): Matrix B is transposed before load
    Returns:
        tir.Call: A handle to the npuir_dot operation
    """

    if size == []:
        A_extent = _get_extent(A)
        B_extent = _get_extent(B)
        C_extent = _get_extent(C)
    else:
        assert len(size) == 3, "size must contains [m, k, n]"
        A_extent = [size[0], size[1]]
        B_extent = [size[1], size[2]]
        C_extent = [size[0], size[2]]

    A = _to_region(A, "r", A_extent)
    B = _to_region(B, "r", B_extent)
    C = _to_region(C, "rw", C_extent)

    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_dot"), A, B, C, initC, a_transpose, b_transpose)


def npuir_load_nd2nz(src, dst, size = []):
    """npuir nd2nz-load data from OUT to L1 at tile-level.

    Args:
        src (Union[tir.Buffer, tir.Var]): Input argument to legalize
        dst (Union[tir.Buffer, tir.Var]): Output argument to legalize
        size ([]): buffer extent
    Returns:
        tir.Call: A handle to the npuir_load_nd2nz operation
    """

    src = _to_region(src, "r", _get_extent(src) if size is [] else size)
    dst = _to_region(dst, "w", _get_extent(dst) if size is [] else size)
    # dst_continuous: whether the source data is stored continuously in the destination buffer.
    # It is good to always set dst_continuous to True.
    dst_continuous = True
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_load_nd2nz"), src, dst, dst_continuous)


def npuir_store_fixpipe(src, dst, size = [], enable_nz2nd = False, channel_split = False, pre_relu_mode = ""):
    """npuir nd2nz-load data from OUT to L1 at tile-level.

    Args:
        src (tir.Buffer): Input argument to legalize
        dst (tir.Buffer): Output argument to legalize
        size ([]): buffer extent
        enable_nz2nd (bool): whether enable nz2nd when store to OUT
        channel_split (bool): whether split channel when store to OUT
        pre_relu_mode (str): "", "relu", "leaky_relu", "prelu"
    Returns:
        tir.Call: A handle to the npuir_store_fixpipe operation
    """

    assert((src.dtype == dst.dtype)
           or (src.dtype == "float32" and dst.dtype == "float16")
           or (src.dtype == "float32" and dst.dtype == "bfloat16")
           or (src.dtype == "int32" and dst.dtype == "int8"),
           "Unexpected pre-quant mode in npuir_store_fixpipe")

    src = _to_region(src, "r", _get_extent(src) if size is [] else size)
    dst = _to_region(dst, "w", _get_extent(dst) if size is [] else size)
    pre_relu_map = {"": 0, "relu": 1, "leaky_relu": 2, "prelu": 3}
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_store_fixpipe"), src, dst,
                           enable_nz2nd, channel_split, pre_relu_map[pre_relu_mode])

def npuir_brc(src, dst):
    """Broadcast a vector or a scalar according to the broadcast axes array

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion, tir.var]): Source vector or scalar
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector

    Raises:
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If input and output shapes do not match for broadcast.

    Returns:
        tir.Call: A handle to the npuir_brc operation
    """
    src_extent = _get_extent(src)
    dst_extent = _get_extent(dst)

    if not isinstance(src, tir.Var):
        assert len(src_extent) == len(
            dst_extent), "The input vector and output vector must have same rank."

        for i in range(0, len(src_extent)):
            if src_extent[i] != 1:
                assert src_extent[i] == dst_extent[
                    i], "The input and output shapes do not match for broadcast."
    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_brc"), src, dst)


def npuir_cast(src, dst, round_mode):
    """Performs element-wise operation on N operands and produces a single result.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        round_mode: Round mode (round/rint/floor/ceil/trunc/odd)

    Raises:
        AssertionError: If input is not vector.
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If round mode is invalid.
        AssertionError: If input and output shapes do not match for broadcast.

    Returns:
        tir.Call: A handle to the npuir_cast operation
    """
    broadcast_dims = []
    valid_round_mode = {"round", "rint", "floor", "ceil", "trunc", "odd"}
    src_extent = _get_extent(src)
    dst_extent = _get_extent(dst)

    assert not isinstance(src, tir.Var), "The first input is vector-only."
    assert len(src_extent) == len(
        dst_extent), "The input/init operands and result have the same rank."
    assert round_mode in valid_round_mode, "Round mode is invalid."

    for i in range(0, len(src_extent)):
        if src_extent[i] != 1:
            assert src_extent[i] == dst_extent[
                i], "The input and output shapes do not match for broadcast."

    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_cast"), src, dst, round_mode)


def npuir_reduce(src, dst, dims:Union[list, tuple], reduce_mode):
    """Reduce one or more axes of the source vector according to the reduction axes array, starting from an init value.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        dims: The reduction indices array
        reduce_mode: Reduce mode (sum/prod/max/min/max_with_index/min_with_index/any/all/xori/ori/none)

    Raises:
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If reduce mode is invalid.
        AssertionError: If The reduction indices array is empty.

    Returns:
        tir.Call: A handle to the npuir_reduce operation
    """
    valid_reduce_mode = {"sum", "prod", "max", "min", "max_with_index", "min_with_index", "any", "all", "xori", "ori", "none"}
    src_extent = _get_extent(src)
    dst_extent = _get_extent(dst)
    assert len(src_extent) == len(
        dst_extent), "The input vector and output vector must have same rank."
    assert reduce_mode in valid_reduce_mode, "Reduce mode is invalid."
    assert len(dims) != 0, "The reduction indices array cannot be empty."

    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)

    reduce_dims = ','.join(str(dim) for dim in dims)
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_reduce"), src, dst, reduce_dims, reduce_mode)

_local = threading.local()

def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "resource_specialize_frame_stack"):
        _local.resource_specialize_frame_stack = FrameStack()
    return _local.resource_specialize_frame_stack


@register_object("tl.ResourceSpecializeFrame")
class ResourceSpecializeFrame(TIRFrame):

    def __enter__(self):
        super().__enter__()
        _get_current_stack().push(self)
        self.name = self.frames[0].attr_key

    def __exit__(self, ptype, value, trace):
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> Optional["ResourceSpecializeFrame"]:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def set(self, other, event_id: int = 0):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_set_flag"), self.name, other, event_id)

    def wait(self, other, event_id: int = 0):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_wait_flag"), other, self.name, event_id)

    def block_barrier(self, id):
        """npuir inter block barrier at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block"), 0, self.name, id)

    def subblock_barrier(self, id):
        """npuir inter subblock barrier at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block"), 1, self.name, id)

    def sync_block_set(self, id):
        """npuir intra block sync at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block_set operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block_set"), 2, self.name, id)

    def sync_block_wait(self, id):
        """npuir intra block sync at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block_wait operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block_wait"), self.name, id)

def ResourceSpecialize(resource: str):
    return _ffi_api.ResourceSpecialize(resource)


rs = ResourceSpecialize


def set_flag(other, event_id: int = 0):
    return ResourceSpecializeFrame.Current().set(other, event_id)


def wait_flag(other, event_id: int = 0):
    return ResourceSpecializeFrame.Current().wait(other, event_id)

def pipe_barrier(pipe):
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_pipe_barrier"), pipe)

def block_barrier(id):
    return ResourceSpecializeFrame.Current().block_barrier(id)

def subblock_barrier(id):
    return ResourceSpecializeFrame.Current().subblock_barrier(id)

def sync_block_set(id):
    return ResourceSpecializeFrame.Current().sync_block_set(id)

def sync_block_wait(id):
    return ResourceSpecializeFrame.Current().sync_block_wait(id)

@register_object("tl.ScopeFrame")
class ScopeFrame(TIRFrame):
    """
    ScopeFrame is a custom TIRFrame that manages mix kernel
    and handles the entry and exit of the kernel launch scope.
    """


def Scope(name):
    """Tools to construct a scope frame.

    Parameters
    ----------
    name : str
        A string representing cube-core or vector-core

    Returns
    -------
        The result ScopeFrame.
    Examples:
        >>> T.Scope("Cube")
    """

    return _ffi_api.Scope(name)