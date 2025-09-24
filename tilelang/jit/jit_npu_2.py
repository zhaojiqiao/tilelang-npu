
  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {{
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {{
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }}
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0]=="*" else "" for i, ty in signature.items()])};
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items())});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__tilelang_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___tilelang_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  {cpp_msprof_callback}
  return m;
}}
"""

def generate_npu_utils_src():
    return """
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "experiment/runtime/runtime/rt.h"

// Use map to differentiate same name functions from different binary
static std::unordered_map<std::string, size_t> registered_names;
static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;

static std::tuple<void *, void *>
registerKernel(const char *name, const void *data, size_t data_size, int shared,
               int device, const char *kernel_mode_str) {
  // name 内核名称
  // data 指向内核2进制的指针
  // data_size 二进制数据的大小
  // shared 未使用
  // device 目标设备ID
  // kernel mod str 内核模式字符串(aiv or others)
  rtError_t rtRet;

  // 创建二进制数据结构
  rtDevBinary_t devbin;
  devbin.data = data;
  devbin.length = data_size;

  // 根据内核模式设置模数
  const std::string kernel_mode{kernel_mode_str};
  if (kernel_mode == "aiv")
    devbin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  else
    devbin.magic = RT_DEV_BINARY_MAGIC_ELF;

  // 设置版本号
  devbin.version = 0;

  rtRet = rtSetDevice(device);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtSetDevice failed, 0x%x\\n", rtRet);
    return {NULL, NULL};
  }

  // 注册二进制数据，获取句柄
  void *devbinHandle = NULL;
  rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtDevBinaryRegister failed, 0x%x\\n", rtRet);
    return {NULL, NULL};
  }

  // 创建唯一的存根名称（避免命名冲突）
  std::string stubName = name;
  stubName += "_" + std::to_string(registered_names[name]);
  registered_names[name]++;

  // 在全局映射并创建存储存根
  auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
  void *func_stub_handle = registered.first->second.get();

  // 注册函数，将二进制句柄与存根关联
  rtRet = rtFunctionRegister(devbinHandle, func_stub_handle, stubName.c_str(),
                             (void *)name, 0);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtFunctionRegister failed(stubName = %s), 0x%x\\n", stubName.c_str(),
           rtRet);
    return {NULL, NULL};
  }

  return std::make_tuple(devbinHandle, func_stub_handle);
}

static PyObject *loadKernelBinary(PyObject *self, PyObject *args) {
  const char *name;        // kernel name
  const char *data;        // binary pointer
  Py_ssize_t data_size;    // binary size
  int shared;              // shared_memory(meaningless now)
  int device;              // device ID
  const char *kernel_mode; // kernel mode

  if (!PyArg_ParseTuple(args, "ss#iis", &name, &data, &data_size, &shared,
                        &device, &kernel_mode)) {
    return NULL;
  }

  auto [module_handle, func_handle] =
      registerKernel(name, data, data_size, shared, device, kernel_mode);

  uint64_t mod = reinterpret_cast<uint64_t>(module_handle);
  uint64_t func = reinterpret_cast<uint64_t>(func_handle);
  if (PyErr_Occurred()) {
    return NULL;
  }

  return Py_BuildValue("(KKii)", mod, func, 0, 0);
}

static PyObject *getArch(PyObject *self, PyObject *args) {
  char name[64] = {'\\0'};

  rtError_t rtRet = rtGetSocVersion(name, 64);

  if (rtRet != RT_ERROR_NONE) {
    printf("rtGetSocVersion failed, 0x%x", rtRet);
    return NULL;
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("s", name);
}

static PyObject *getAiCoreNum(PyObject *self, PyObject *args) {
  uint32_t aiCoreCnt;

  rtError_t rtRet = rtGetAiCoreCount(&aiCoreCnt);

  if (rtRet != RT_ERROR_NONE) {
    printf("rtGetAiCoreCount failed, 0x%x", rtRet);
    return NULL;
  }
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("I", aiCoreCnt);
}

static PyMethodDef NpuUtilsMethods[] = {
    {"load_kernel_binary", loadKernelBinary, METH_VARARGS,
     "Load NPU kernel binary into NPU driver"},
    {"get_arch", getArch, METH_VARARGS, "Get soc version of NPU"},
    // sentinel
    {"get_aicore_num", getAiCoreNum, METH_VARARGS, "Get the number of AI core"},
    {NULL, NULL, 0, NULL}};

static PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT, "npu_utils",
    "Utilities for fetching NPU device info and preparing kernel binary", -1,
    NpuUtilsMethods};

PyMODINIT_FUNC PyInit_npu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, NpuUtilsMethods);
  return m;
}
"""

def read_binary_file(file_path, mode='rb', chunk_size=None, return_type='bytes'):
    """
    读取二进制文件的函数
    
    参数:
        file_path (str): 要读取的文件路径
        mode (str): 文件打开模式，默认为'rb'（二进制读取）
        chunk_size (int): 如果指定，则以指定大小的块读取文件；否则读取整个文件
        return_type (str): 返回数据类型，可以是'bytes'或'bytearray'
    
    返回:
        根据return_type参数返回bytes或bytearray对象
        如果指定了chunk_size，则返回一个生成器，逐块产生数据
    
    异常:
        FileNotFoundError: 当文件不存在时
        IOError: 当读取文件发生错误时
    """
    try:
        with open(file_path, mode) as file:
            if chunk_size:
                # 分块读取文件
                def chunk_reader():
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        if return_type == 'bytearray':
                            yield bytearray(chunk)
                        else:
                            yield chunk
                return chunk_reader()
            else:
                # 一次性读取整个文件
                data = file.read()
                if return_type == 'bytearray':
                    return bytearray(data)
                else:
                    return data
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except IOError as e:
        raise IOError(f"读取文件时发生错误: {e}")

class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # cache_path = "/host/gxy/tilelang-ascend/npu_example/mlir_files/minicv_jit/npu_utils.so"
        # 3 important path
        cache_path = "npu_utils.so"
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device):
        fnname, mix_mode = name.split()
        return self.npu_utils_mod.load_kernel_binary(fnname, kernel, shared, device, mix_mode)

    @functools.lru_cache()
    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid tilelang-compiler complain
        # fetch available memory at runtime
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_aicore_num()

    @functools.lru_cache()
    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]

class JitKernel_NPU:
    def __init__(self, metadata : dict) -> None:
        # 1 launch path
        self.so_launcher_path = f"{metadata['kernel_name']}.so"
        self.utils_name = f"{metadata['name']} aiv"
        # 2 kernel path
        self.utils_kernel_src = read_binary_file(f"{metadata['kernel_name']}.o")
        self.utils_shared = 1
        self.utils_device = torch.npu.current_device()
        self.launch_stream = torch.npu.current_stream(torch.npu.current_device()).npu_stream
        self.launch_grid = [32, 1, 1]
        self.launch_packedMetadata = {"kernel_name":f"{metadata['name']}", "tensor_kinds":metadata['tensor_kinds']}
        self.launch_metadata = {}  # Python对象
        self.launch_enter_hook = None  # Python对象
        self.launch_exit_hook = None  # Python对象
        # self._launch()

    def _launch(self) :
        import importlib.util
        spec = importlib.util.spec_from_file_location("__tilelang_launcher", self.so_launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch_npu = getattr(mod, "launch")

    def __call__(self, *args: Any) -> Any:
        self._launch()
        npu_utils = NPUUtils()
        t_module, t_function, t_n_regs, t_n_spills = npu_utils.load_binary(
            self.utils_name, self.utils_kernel_src, self.utils_shared, self.utils_device)
        return self.launch_npu(self.launch_grid[0], self.launch_grid[1], self.launch_grid[2],
         self.launch_stream, t_function, self.launch_packedMetadata, 
         self.launch_metadata, self.launch_enter_hook, self.launch_exit_hook,
         *args)

class compiler_npu:
    def __init__(self) -> None:
        pass

    def compile(self, mlir_path : str) -> JitKernel_NPU:
        self.metadata = {"shared" : 1}
        if mlir_path.endswith(".mlir") :
            self.mlir_content = self._read_mlir_file(mlir_path)
        else:
            self.mlir_content = mlir_path
        self.constants = {}
        self.signature = self._parse_signature()
        self.workspace_size = -1
        self.mix_mode = "aiv"
        self.lock_num = -1
        self.lock_ini_val = 0
        self._parse_linalg_metadata()
        self._linalg_to_bin_enable_npu_compile()
        self.wrapper_src = generate_npu_wrapper_src(self.constants, 
            self.signature, self.workspace_size, self.mix_mode, self.lock_num, self.lock_ini_val)
        self.so_launcher_path = self.make_npu_launcher_stub(self.metadata['kernel_name'], self.wrapper_src)
        self.wrapper_utiles = generate_npu_utils_src()
        self.so_utils_path = self.make_npu_launcher_stub("npu_utils", self.wrapper_utiles)
        return JitKernel_NPU(metadata=self.metadata)

    def _read_mlir_file(self, file_path) -> str:
        """
        读取MLIR文件内容并返回字符串
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 不存在")
            return None
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return None
    
    def _parse_linalg_metadata(self) -> None:
        """
        Parse Linalg IR to extract metadata required for NPU compilation.
        Extracts and updates the following fields in metadata:
          - mix_mode
          - kernel_name
          - tensor_kinds
          - shared (currently hardcoded)
          - name (combined kernel_name and mix_mode)

        Additionally, removes the mix_mode attribute from the IR.
        """
        # --- Regular expressions and examples ---

        # Example: mix_mode = "aiv" -> aiv
        MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'

        # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
        KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

        # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
        TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

        # Example removal:   ', mix_mode = "aiv"' → ''
        REMOVE_MIX_MODE_REGEX = r', mix_mode\s*=\s*"[^"]*"'

        # Note: Compiled Kernel requires to estimate size of shared memory to occupy
        # Currently, NPU backend does not limit on shared memory
        self.metadata['shared'] = 1
        # the mix mode is also encoded into metadata['name'] for runtime to distinguish
        kernel_name = re.search(KERNEL_NAME_REGEX, self.mlir_content).group(1)
        self.metadata['kernel_name'] = kernel_name

        if kernel_name and '_' in kernel_name:
            self.metadata['name'] = kernel_name.split('_')[0]
        else:
            self.metadata['name'] = kernel_name

        # Parse all tensor kinds from arguments
        self.metadata['tensor_kinds'] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, self.mlir_content)]
        # remove the mix_mode attribute
        self.mlir_content = re.sub(REMOVE_MIX_MODE_REGEX, "", self.mlir_content)

    def _parse_signature(self) -> dict:
        """
        从MLIR文本中解析参数类型并返回字典
        """
        # 定义关心的数据类型
        target_types = {"i1", "i8", "i16", "i32", "i64", "u32", "u64", 
                       "fp16", "bf16", "fp32", "f32", "fp64", "f16"}

        # 提取函数签名部分（括号内的内容）
        pattern = r'func\.func\s*@[^(]*\(([^)]*)\)'
        match = re.search(pattern, self.mlir_content)

        if not match:
            return {}

        params_str = match.group(1)

        # 分割参数
        params = []
        current_param = ""
        brace_count = 0
        angle_count = 0

        for char in params_str:
            if char == ',' and brace_count == 0 and angle_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '<':
                    angle_count += 1
                elif char == '>':
                    angle_count -= 1

        if current_param:
            params.append(current_param.strip())

        result = {}
        index = 0

        for param in params:
            # 跳过以%arg开头的参数
            if re.match(r'%arg\d+', param.strip()):
                continue
            
            # 检查类型是否包含目标类型
            found_type = None
            for t_type in target_types:
                # 检查是否有x前缀的类型（如xf16）
                x_pattern = r'\bx' + t_type + r'\b'
                if re.search(x_pattern, param):
                    found_type = '*' + t_type
                    break
                # 检查普通类型（如i32）
                elif re.search(r'\b' + t_type + r'\b', param):
                    found_type = t_type
                    break

            if found_type:
                # 特殊处理：f16应映射为fp16
                if found_type == 'f16':
                    found_type = 'fp16'
                elif found_type == '*f16':
                    found_type = '*fp16'

                result[index] = found_type
                index += 1

        return result

    def _linalg_to_bin_enable_npu_compile(self):
        linalg = self.mlir_content
        metadata = self.metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
            Path(ttadapter_path).write_text(linalg)
            bin_file = os.path.join(tmpdir, "kernel")
            if self._check_bishengir_api_change():
                bin_file_with_ext = "kernel.o"
            else:
                bin_file_with_ext = "kernel_reloc.o"
            if self._check_bishengir_is_regbased():
