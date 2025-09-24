# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The compiler for TL programs."""

import os
import os.path as osp
from typing import Union, Optional, Callable, List
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tir
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang.contrib import hipcc, nvcc
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.utils.target import determine_target  # noqa: F401
from tilelang.engine.phase import (
    LowerAndLegalize,
    OptimizeForTarget,
)


def is_cpu_device_backend(target: Target):
    return target.kind.name == "c"


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and
                attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tir.PrimFunc):
    attrs = func.attrs

    # Check if it's a C target
    if "target" in attrs and attrs["target"].kind.name == "c":
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tir.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


@tvm.register_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target):
    project_root = osp.join(osp.dirname(__file__), "../..")
    if "TL_TEMPLATE_PATH" in os.environ:
        tl_template_path = os.environ["TL_TEMPLATE_PATH"]
    else:
        tl_template_path = osp.abspath(osp.join(project_root, "src"))
    # TODO(lei): this indeed should be renamed into
    # TL_CUTLASS_INCLUDE_PATH in the future
    if "TL_CUTLASS_PATH" in os.environ:
        cutlass_path = os.environ["TL_CUTLASS_PATH"]
    else:
        cutlass_path = osp.abspath(osp.join(project_root, "3rdparty/cutlass/include"))
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))

    # special handle for Hopper
    if compute_version == "90":
        arch = ["-arch=sm_90a"]
        format = "cubin"
    else:
        arch = [f"-arch=sm_{compute_version}"]
        format = "cubin"

    # printing out number of registers
    debug_option = "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
    ptx = nvcc.compile_cuda(
        code,
        format,
        arch,
        options=[
            "-std=c++17",
            debug_option,
            "--use_fast_math",
            "-I" + tl_template_path,
            "-I" + cutlass_path,
        ],
        verbose=False,
    )

    return ptx


@tvm.register_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    project_root = osp.join(osp.dirname(__file__), "../..")
    tl_template_path = osp.abspath(osp.join(project_root, "src"))

    # TODO(lei): actually this indeed should be renamed into
    # TL_COMPOSABLE_KERNEL_INCLUDE_PATH in the future
    if "TL_COMPOSABLE_KERNEL_PATH" in os.environ:
        ck_path = os.environ["TL_COMPOSABLE_KERNEL_PATH"]
    else:
        ck_path = osp.abspath(osp.join(project_root, "3rdparty/composable_kernel/include"))

    hsaco = hipcc.compile_hip(
        code,
        target_format="hsaco",
        options=[
            "-std=c++17",
            "-I" + tl_template_path,
            "-I" + ck_path,
        ],
        verbose=False,
    )

    return hsaco


def extrac_params(func: tir.PrimFunc) -> List[KernelParam]:
    tensor_types = []
    for var in func.params:
        if var in func.buffer_map:
            tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
        else:
            tensor_types.append(KernelParam.from_var(var))
    return tensor_types


def canon_target_host(target: Union[str, Target], target_host: Optional[Union[str, Target]]):

    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    return target_host


def host_codegen(host_mod: tvm.IRModule, target_host: Target) -> tvm.IRModule:
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tir.transform.LowerIntrin()(host_mod)
    host_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    if target_host.kind.name == "llvm":
        host_mod = tvm._ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    elif target_host.kind.name == "c":
        host_mod = tvm._ffi.get_global_func("target.build.c")(host_mod, target_host)
    else:
        raise ValueError(f"Target host {target_host.kind.name} is not supported")
    return host_mod


def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    if target.kind.name == "npuir":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_npuir")(device_mod, target)
        return device_mod
    # device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    # device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)

    # if target.kind.name == "cuda":
    #     device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda")(device_mod, target)
    # elif target.kind.name == "hip":
    #     device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    # elif target.kind.name == "ascend":
    if True:
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_ascend")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    if target.kind.name == "cuda":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
            device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip_without_compile")(
            device_mod, target)
    elif target.kind.name == "c":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cpp")(device_mod, target)
    elif target.kind.name == "llvm":
        device_mod = tvm._ffi.get_global_func("target.build.llvm")(device_mod, target)
    elif target.kind.name == "webgpu":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_webgpu")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def lower(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
    '''
        enable_host_codegen: whether to enable host codegen, default is False, as we have our
        own host codegen implementation in jit.
        enable_device_compile: whether to enable device codegen, default is False, as we have our
        own device codegen implementation in jit.
    '''

    mod = func_or_mod
    params = None
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        params = extrac_params(func) if not runtime_only else None
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    # _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    # _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))
    # print(target.kind.name)
    # Phase 1: Lower and legalize the IR
    mod = LowerAndLegalize(mod, target)

    # Phase 2: Optimize the IR for the target
    mod = OptimizeForTarget(mod, target)

    codegen_mod = device_codegen(mod, target)
    # print(codegen_mod.get_source())
    return codegen_mod.get_source()
    # if enable_host_codegen:
    #     host_mod = host_codegen(host_mod, target_host)
    #     host_mod.import_module(codegen_mod)
    #     return CompiledArtifact(
    #         host_mod, device_mod, params, codegen_mod.get_source(), rt_mod=host_mod)

    return CompiledArtifact(None, None, params, codegen_mod.get_source())
