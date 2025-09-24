import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union
import shutil
import sysconfig
import pybind11
import torch
import torch_npu
import functools
import sys

import pickle
from contextlib import contextmanager

def _get_npucompiler_path() -> str:
    # 设置编译器的环境变量
    os.environ["BISHENG_INSTALL_PATH"] = "/host/zyz/tilelang-ascend/extra_tools"
    return "/host/zyz/tilelang-ascend/extra_tools/bishengir-compile"

def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12, # BOOL
        # Integer types
        "i8": 2, # INT8
        "i16": 6, # INT16
        "i32": 3, # INT32
        "i64": 9, # INT64
        # Unsigned integer types
        "u32": 8, # UINT32
        "u64": 10, # UINT64
        # Floating point types
        "fp16": 1, # FLOAT16
        "bf16": 27, # DT_BF16
        "fp32": 0, # FLOAT
        "fp64": 11, # DOUBLE
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")
    
    return MAP_SIGTYPE_TO_INT[sigty]

def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bisheng or TRITON_NPU_COMPILER_PATH"
            )
            bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path

def extract_device_print_code_from_cann():
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib")

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), 'r') as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = ' '.join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = '\n'.join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer('\[aicore\]', code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos-1, -1, -1):
                if src[i] == '}': # this relies on that all [aicore] functions come after normal funcitons
                    left = i+1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == '{':
                    brace_nest += 1
                elif src[j] == '}':
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ''
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += codep[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace('__gm__', ' ')
        new_code = new_code.replace('__CCELIB_RT_ERROR_NONE', 'RT_ERROR_NONE')
        new_code = new_code.replace('__CCELIB_RT_MEMORY_HBM', 'RT_MEMORY_HBM')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_HOST_TO_DEVICE', 'RT_MEMCPY_HOST_TO_DEVICE')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_DEVICE_TO_HOST', 'RT_MEMCPY_DEVICE_TO_HOST')
        return new_code

def generate_npu_wrapper_src(constants, signature, workspace_size, mix_mode, lock_num, lock_ini_val):
    def _ty_to_cpp(ty):
        if ty[0] == '*':
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _extracted_ty(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            "i1": "int32_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gradX, gridY, gridZ;
        rtStream_t stream;
        const void *function;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])

    grid_info = {'X':'i32', 'Y':'i32', 'Z':'i32'}

    enable_device_print = os.getenv(
        "TRITON_DEVICE_PRINT", 'false').lower().in ('true', '1')
    enable_taskqueue = os.getenv(
        "TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
    # enable_auto_map_parallel_blocks = _is_auto_map_parallel_blocks_enabled()
    enable_auto_map_parallel_blocks = False
    # npu_utils = NPUUtils()
    # num_physical_blocks = npu_utils.get_aivector_core_num(
    # ) if mix_mode == "aiv" else npu_utils.get_aicore_num()
    num_physical_blocks = 48
    task_type = "MSPROF_GE_TASK_TYPE_AIV" if mix_mode == "aiv" else "MSPROF_GE_TASK_TYPE_AI_CORE"
    LINE_CHANGE_CHAR = chr(10) # it is \n

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsingedLongLong(ret));
    if (!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret); // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsinged int len);
  extern int MsprofReportApi(unsinged int agingFlag, const MsprofApi *api);
  extern unsinged long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsinged int moduleId, callback handle);
  static unsigned int __MsprofFlagL0 = 0;
  static unsinged int __MsprofFlagL1 = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  MSprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENMSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {LINE_CHANGE_CHAR.join(
          f'dataTypes[{i}] = {convert_sigtype_to_int(ty[1:])};'
          for i, ty in signature.items()
          if ty.startswith("*") and i < 5
        )}
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; J++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // Default tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
            fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
            tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
            fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
            tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string>
#include <sys/syscall.h>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
{'#include <torch_npu/csrc/framework/OpCommand.h>' if enable_taskqueue else ''}
#include "experiment/runtime/runtime/rt.h"
{extract_device_print_code_from_cann() if enable_device_print else ''}

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds, {arg_decls}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  {'auto launch_call = [=]()' if enable_taskqueue else ''} {{
    uint32_t blockNum = gridX*gridY*gridZ;
    {'blockNum = std::min(blockNum, (uint32_t)' + str(num_physical_blocks) + ');' if enable_auto_map_parallel_blocks else ''}
    {'cce::internel::DebugTunnelData *DTData = cce::internel::DebugTunnel::Open(blockNum);' if enable_device_print else ''}
    rtError_t ret;
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {{
        return {'ret' if enable_taskqueue else ''};
    }}
    // stub argument for workspace
    void *syncBlockLock = NULL;
    void *workSpace_addr = NULL;
    uint16_t ModuleId = 0;
    {f'''
    uint64_t syncBlockLockSize = {lock_num} * sizeof(int64_t);
    ret = rtMalloc(reinterpret_cast<void **>(&syncBlockLock),
                   syncBlockLockSize, RT_MEMORY_HBM, 0);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    std::vector<int64_t> lockInitData({lock_num}, {lock_ini_val});
    ret = rtMemcpy(syncBlockLock, syncBlockLockSize, reinterpret_cast<void *>(lockInitData.data()),
                   syncBlockLockSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (ret != RT_ERROR_NONE) {{
        return {'ret' if enable_taskqueue else ''};
    }}
    ''' if lock_num > 0 else ''}
    {f'''
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    ret = rtMalloc(reinterpret_cast<void **>(&workspace_addr),
                   totalWorkSpaceSize, RT_MEMORY_HBM, ModuleId);
    if (ret != RT_ERROR_NONE) {{
        return {'ret' if enable_taskqueue else ''};
    }}
    ''' if workspace_size > 0 else ''}
    struct __attribute__((packed)) {{
      void* ffts_addr __attribute__((aligned(8)));
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants)}
      {' '.join(f'{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
      {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
    }} args = {{
        static_cast<void*>(ffts_addr),
        static_cast<void*>(syncBlockLock),
        static_cast<void*>(workspace_addr),
        {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants)},
        {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
        {', static_cast<void*>(DTData)' if enable_device_print else ''}
    }};
    {cpp_msprof_call_before_launch}
    ret = rtKernelLaunch(func, blockNum, sttaic_cast<void*>(&args), sizeof(args), NULL, stream);
    {'void *&stream_ref = const_cast<void*&>(stream);' if enable_device_print else ''}
    {'cce::internel::DebugTunnel::Close(DTData, stream_ref);' if enable_device_print else ''}
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else ''}
    }};
    {'at_npu::native::OpCommand cmd; cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();' if enable_taskqueue else ''}
    return ;
}}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_type(PyObject *tensor) {{
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor==Py_None) {{
    return shape;
  }}

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {{
    return shape;
  }}
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {{
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject* items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {{
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {{
        shape.push_back(PyLong_AsLong(dim));
      }}
    }}
  }}
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}
  if (__MsprofFlagL1)
  {{
    {
      LINE_CHANGE_CHAR.JOIN(
        f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
        for i, ty in signature.items() if ty[0] == "*"
      )
    }
  }}