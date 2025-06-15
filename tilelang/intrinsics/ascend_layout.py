# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tvm import arith, DataType
import tilelang.language as T

BYTE_PER_C0 = 32
C0_NUM_PER_FRACTAL = 16
BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL


def ceil_div(a, b):
    """Ceiling division of a by b."""
    return (a + b - 1) // b


def round_up(a, b):
    """Round up a to the nearest multiple of b."""
    return ceil_div(a, b) * b


def make_zn_layout(buf):
    ana = arith.Analyzer()
    dtype = buf.dtype
    shape = buf.shape

    ELE_NUM_PER_C0 = BYTE_PER_C0 // DataType(dtype).bits * 8
    ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL // DataType(dtype).bits * 8

    def transform_func(*args):
        i, j = args[-2:]
        """
        zn:
            Layout: 
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(rows)),
                          MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv<ELE_NUM_PER_C0>(cols))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                           MakeStride(Int<1>{}, (int64_t)RoundUp<Catlass::C0_NUM_PER_FRACTAL>(rows) * ELE_NUM_PER_C0)));

            Coord compute:
                const uint32_t rowsInFractal = get<0, 0>(shape);
                const uint32_t colsInFractal = get<1, 0>(shape);
                const int64_t strideRowsByFractal = get<0, 1>(stride);
                const int64_t strideColsByFractal = get<1, 1>(stride);
                return row / rowsInFractal * strideRowsByFractal + col / colsInFractal * strideColsByFractal
                    + (row % rowsInFractal) * get<0, 0>(stride) + (col % colsInFractal) * get<1, 0>(stride);        
        """
        new_shape = [[C0_NUM_PER_FRACTAL,
                      ceil_div(shape[-2], C0_NUM_PER_FRACTAL)],
                     [ELE_NUM_PER_C0, ceil_div(shape[-1], ELE_NUM_PER_C0)]]
        new_stride = [[ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL],
                      [1, round_up(shape[-2], C0_NUM_PER_FRACTAL) * ELE_NUM_PER_C0]]
        rowInFractal = new_shape[0][0]
        colInFractal = new_shape[1][0]
        strideRowsByFractal = new_stride[0][1]
        strideColsByFractal = new_stride[1][1]

        return [
            *args[:-2],
            ana.simplify(i // rowInFractal * strideRowsByFractal +
                         j // colInFractal * strideColsByFractal +
                         (i % rowInFractal) * new_stride[0][0] +
                         (j % colInFractal) * new_stride[1][0])
        ]

    return T.Layout(shape, transform_func)
