tilelang-npu User Guide:

To install tilelang-npu, simply run `bash install_ascend.sh`. Alternatively, you can refer to this script to manually build the dynamic library.

Inside the `npu_example` directory:

    + `examples`: Contains a pre-generated AscendC code example. You can compile it directly using `build.sh`. The resulting dynamic library can then be used by running `python test.py`.
    + `npu_ws.py`: Matrix multiplication example. Run `python npu_ws.py` to generate the corresponding C++ code.

