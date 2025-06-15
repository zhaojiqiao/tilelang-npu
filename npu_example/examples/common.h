#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace tl::ascend {
using namespace Catlass;
using namespace tla;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Gemm::Block;
using namespace Catlass::Epilogue::Tile;
using namespace AscendC;

using ArchTag = Arch::AtlasA2;
using LayoutL1 = layout::zN;
using LayoutGM = layout::RowMajor;

using LayoutL0A = layout::zZ;
using LayoutL0B = layout::nZ;

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_l1(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor) {
  auto layout = MakeLayoutFromTag(LayoutGM{srcM, srcN});
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                             AscendC::TPosition::GM>(srcTensor, src_LAYOUT);

  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layoutInL1 = tla::MakeLayout<T, LayoutL1_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutInL1),
                             AscendC::TPosition::A1>(dstTensor, layoutInL1);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_l1_to_l0a(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor) {
  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layout = tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));

  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0A_ = Catlass::detail::TagToLayout_t<T, LayoutL0A>;
  constexpr auto layoutAInL0 = tla::MakeLayout<T, LayoutL0A_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutAInL0),
                             AscendC::TPosition::A2>(dstTensor, layoutAInL0);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_l1_to_l0b(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor) {
  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto LAYOUT = tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(LAYOUT, tla::MakeShape(dstM, dstN));
  ;

  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0B_ = Catlass::detail::TagToLayout_t<T, LayoutL0B>;
  constexpr auto layoutBInL0 = tla::MakeLayout<T, LayoutL0B_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutBInL0),
                             AscendC::TPosition::B2>(dstTensor, layoutBInL0);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool init>
CATLASS_DEVICE void mma(LocalTensor<T1> A, LocalTensor<T1> B, LocalTensor<T2> C,
                        uint8_t unitFlag = 0) {
  MmadParams mmadParams;
  mmadParams.m = M;
  mmadParams.n = N;
  mmadParams.k = K;
  mmadParams.cmatrixInitVal = init;
  // mmadParams.unitFlag = unitFlag;

  Mmad(C, A, B, mmadParams);

  constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
  if constexpr ((M / C0_NUM_PER_FRACTAL) * (N / C0_NUM_PER_FRACTAL) <
                PIPE_M_BARRIER_THRESHOLD) {
    PipeBarrier<PIPE_M>();
  }
}

template <typename T1, typename T2, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_l0c_to_gm(GlobalTensor<T2> dstTensor,
                                   LocalTensor<T1> srcTensor,
                                   uint8_t unitFlag = 0) {
  auto layoutInL0C = tla::MakeLayoutL0C(srcM, srcN); // TODO (xwh): round up?
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(layoutInL0C),
                             AscendC::TPosition::CO1>(srcTensor, layoutInL0C);
  LayoutGM gm{dstM, dstN};
  auto layout = MakeLayoutFromTag(gm);
  auto dTensor = MakeTensor(dstTensor, layout, Arch::PositionGM{});
  auto layout_ = dTensor.layout();
  auto dst_LAYOUT = MakeLayoutTile(layout_, tla::MakeShape(srcM, srcN));
  auto dst = MakeTensor<decltype(dstTensor), decltype(dst_LAYOUT),
                        AscendC::TPosition::GM>(dstTensor, dst_LAYOUT);

  CopyL0CToGmTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src, unitFlag);
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t block_M,
          uint32_t block_N, uint32_t SwizzleOffset = 1,
          uint32_t SwizzleDirection = 0>
CATLASS_DEVICE auto thread_block_swizzle(uint64_t pid) {
  GemmCoord problem_shape = GemmCoord(M, N, K);
  MatrixCoord tile_shape = MatrixCoord(block_M, block_N);

  GemmIdentityBlockSwizzle swizzle =
      GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>(problem_shape,
                                                                tile_shape);

  auto cols = swizzle.loopsMN.column();

  auto coord = swizzle.GetBlockCoord(pid);

  return coord.m() * cols + coord.n();
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_ub(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor) {
  using LayoutSrc = layout::RowMajor;
  using LayoutDst = layout::RowMajor;

  using GType = Gemm::GemmType<T, layout::RowMajor>;

  auto copy_ = CopyGm2Ub<ArchTag, GType>();

  copy_(dstTensor, srcTensor, LayoutDst{dstM, dstN},
        LayoutSrc{dstM, dstN, srcN}); // row, col, ldm
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_ub_to_gm(GlobalTensor<T> dstTensor,
                                  LocalTensor<T> srcTensor) {
  using LayoutDst = layout::RowMajor;
  using LayoutSrc = layout::RowMajor;

  using GType = Gemm::GemmType<T, layout::RowMajor>;
  auto copy_ = CopyUb2Gm<ArchTag, GType>();
  copy_(dstTensor, srcTensor, LayoutDst{srcM, srcN, dstN},
        LayoutSrc{srcM, srcN});
}

template <typename T, uint32_t Len>
CATLASS_DEVICE void tile_add(LocalTensor<T> const &ubIn0,
                             LocalTensor<T> const &ubIn1,
                             LocalTensor<T> const &ubOut) {
  AscendC::Add(ubOut, ubIn0, ubIn1, Len);
}

} // namespace tl::ascend