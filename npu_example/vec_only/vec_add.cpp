#include "common.h"
#include "acl/acl.h"
using namespace Catlass;

CATLASS_GLOBAL
void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half*)A_handle);
  AscendC::GlobalTensor<half> B;
  B.SetGlobalBuffer((__gm__ half*)B_handle);
  AscendC::GlobalTensor<half> C;
  C.SetGlobalBuffer((__gm__ half*)C_handle);

  auto cid = AscendC::GetBlockIdx();
  AscendC::TBuf<AscendC::TPosition::VECIN> A_VEC;
  pipe.InitBuffer(A_VEC, 32768 * 2);
  AscendC::TBuf<AscendC::TPosition::VECIN> B_VEC;
  pipe.InitBuffer(B_VEC, 32768 * 2);
  AscendC::TBuf<AscendC::TPosition::VECIN> C_VEC;
  pipe.InitBuffer(C_VEC, 32768 * 2);
  pipe.Destroy();

  auto vid = AscendC::GetSubBlockIdx();
  {
    auto A_VEC_ = A_VEC.Get<half>();
    tl::ascend::copy_gm_to_ub<half, 1024, 1024, 128, 256>(A_VEC_[0], A[(((cid / 4) * 131072) + ((cid % 4) * 256))]);
  }
  {
    auto B_VEC_ = B_VEC.Get<half>();
    tl::ascend::copy_gm_to_ub<half, 1024, 1024, 128, 256>(B_VEC_[0], B[(((cid / 4) * 131072) + ((cid % 4) * 256))]);
  }
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
  {
    auto A_VEC_ = A_VEC.Get<half>();
    auto B_VEC_ = B_VEC.Get<half>();
    auto C_VEC_ = C_VEC.Get<half>();
    tl::ascend::tile_add<half, 32768>(A_VEC_[0],B_VEC_[0],C_VEC_[0]);
  }
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
  {
    auto C_VEC_ = C_VEC.Get<half>();
    tl::ascend::copy_ub_to_gm<half, 128, 256, 1024, 1024>(C[(((cid / 4) * 131072) + ((cid % 4) * 256))], C_VEC_[0]);
  }
}

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream) {
  main_kernel<<<32, nullptr, stream>>>(A_handle, B_handle, C_handle);
}