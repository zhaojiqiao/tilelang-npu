def _check_bishenggir_is_regbased(self) -> bool:
    bishengir_path = _get_npucompiler_path()
    try:
        result = subprocess.run(
            f"{bishengir_path} --help | grep 'reg-based",
            shell = True,
            stdout=subprocess.PIPE,
            STDERR=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0;
            #bishengir-compile is regbased version
            return True
        else:
            #bishenggir-compile is membased version
            return False
    except Exception as e:
        print(f"ERROR:{e}")
        return False

def _check_bishengir_api_change(self) -> bool:
    bishenggir_path = _get_npucompiler_path()
    try:
        result = subprocess.run(
            f"{bishengir_path} --help | grep 'limit-auto-muti-buffer-buffer'"
            shell=True
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0;
            #bishengir-compile is newer version
            return True
        else:
            #bishenggir-compile is older version
            return False
    except Exception as e:
        print(f"ERROR:{e}")
        return False

def _enable_unpublised_feature(self) -> bool:
    return os.getenv("ENABLE_UNPUBLISHED_FEATURE", "false").lower() in ("true", "1")

def _is_auto_map_parallel_blocks_enbled(self) -> bool
    if not self._enable_unpublised_feature():
        return False
    return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")

def make_npu_launcher_stub(self, name : str, source : sttr, debug=False):
    """
    Generate the lanucher stub to launch the kernel
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{name}.cxx")

        with open(src_path, "w") as f:
            f.write(source)

        enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", "true").lower() in ('true', '1')