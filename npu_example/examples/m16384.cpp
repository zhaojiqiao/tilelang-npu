#include "acl/acl.h"
#include "common.h"
using namespace Catlass;

CATLASS_GLOBAL
void main_kernel(GM_ADDR A_handle, GM_ADDR B_handle, GM_ADDR C_handle) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half *)A_handle);
  AscendC::GlobalTensor<half> B;
  B.SetGlobalBuffer((__gm__ half *)B_handle);
  AscendC::GlobalTensor<half> C;
  C.SetGlobalBuffer((__gm__ half *)C_handle);

  auto cid = AscendC::GetBlockIdx();
  AscendC::TBuf<AscendC::TPosition::CO1> C_L0;
  pipe.InitBuffer(C_L0, 32768 * 4);
  AscendC::TBuf<AscendC::TPosition::A1> A_L1;
  pipe.InitBuffer(A_L1, 65536 * 2);
  AscendC::TBuf<AscendC::TPosition::A1> B_L1;
  pipe.InitBuffer(B_L1, 131072 * 2);
  AscendC::TBuf<AscendC::TPosition::A2> A_L0;
  pipe.InitBuffer(A_L0, 16384 * 2);
  AscendC::TBuf<AscendC::TPosition::B2> B_L0;
  pipe.InitBuffer(B_L0, 32768 * 2);
  pipe.Destroy();

  cid = tl::ascend::thread_block_swizzle<16384, 16384, 16384, 128, 256, 3, 0>(
      cid);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(0);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(1);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
  for (int k = 0; k < 64; ++k) {
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>((k % 2));
    {
      auto A_L1_ = A_L1.Get<half>();
      tl::ascend::copy_gm_to_l1<half, layout::zN, 16384, 16384, 128, 256>(
          A_L1_[((k % 2) * 32768)], A[(((cid / 64) * 2097152) + (k * 256))]);
    }
    {
      auto B_L1_ = B_L1.Get<half>();
      tl::ascend::copy_gm_to_l1<half, layout::zN, 16384, 16384, 256, 256>(
          B_L1_[((k % 2) * 65536)], B[((k * 4194304) + ((cid % 64) * 256))]);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((k % 2));
    for (int kk = 0; kk < 4; ++kk) {
      if (kk == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((k % 2));
      }
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((kk % 2));
      {
        auto A_L1_ = A_L1.Get<half>();
        auto A_L0_ = A_L0.Get<half>();
        tl::ascend::copy_l1_to_l0a<half, layout::zN, 128, 256, 128, 64>(
            A_L0_[((kk % 2) * 8192)], A_L1_[(((k % 2) * 32768) + (kk * 8192))]);
      }
      {
        auto B_L1_ = B_L1.Get<half>();
        auto B_L0_ = B_L0.Get<half>();
        tl::ascend::copy_l1_to_l0b<half, layout::zN, 256, 256, 64, 256>(
            B_L0_[((kk % 2) * 16384)],
            B_L1_[(((k % 2) * 65536) + (kk * 1024))]);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((kk % 2));
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((kk % 2));
      if ((k == 0) && (kk == 0)) {
        {
          auto A_L0_ = A_L0.Get<half>();
          auto B_L0_ = B_L0.Get<half>();
          auto C_L0_ = C_L0.Get<float>();
          tl::ascend::mma<half, float, 128, 256, 64, true>(
              A_L0_[(kk * 8192)], B_L0_[(kk * 16384)], C_L0_[0]);
        }
      }
      if ((0 < k) || (0 < kk)) {
        {
          auto A_L0_ = A_L0.Get<half>();
          auto B_L0_ = B_L0.Get<half>();
          auto C_L0_ = C_L0.Get<float>();
          tl::ascend::mma<half, float, 128, 256, 64, false>(
              A_L0_[((kk % 2) * 8192)], B_L0_[((kk % 2) * 16384)], C_L0_[0]);
        }
      }
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((kk % 2));
      if (kk == 3) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>((k % 2));
        if (k == 63) {
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        }
      }
    }
  }
  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
  {
    auto C_L0_ = C_L0.Get<float>();
    tl::ascend::copy_l0c_to_gm<float, half, 128, 256, 16384, 16384>(
        C[(((cid / 64) * 2097152) + ((cid % 64) * 256))], C_L0_[0]);
  }
  AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(0);
  AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(1);
  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, uint8_t *C_handle,
                     aclrtStream stream) {
  main_kernel<<<8192, nullptr, stream>>>(A_handle, B_handle, C_handle);
}