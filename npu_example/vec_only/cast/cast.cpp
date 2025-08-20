#include "common.h"
#include "acl/acl.h"
using namespace Catlass;
using namespace AscendC;

CATLASS_GLOBAL
void main_kernel( GM_ADDR INPUT_handle,  GM_ADDR OUTPUT_handle) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> INPUT;
  INPUT.SetGlobalBuffer((__gm__ float*)INPUT_handle);
  AscendC::GlobalTensor<half> OUTPUT;
  OUTPUT.SetGlobalBuffer((__gm__ half*)OUTPUT_handle);

  auto cid = AscendC::GetBlockIdx();
  AscendC::TBuf<AscendC::TPosition::VECIN> ubIn;
  pipe.InitBuffer(ubIn, 16384 * 4);
  AscendC::TBuf<AscendC::TPosition::VECIN> ubOut;
  pipe.InitBuffer(ubOut, 16384 * 2);

  pipe.Destroy();

  {
    auto ubIn_ = ubIn.Get<float>();
    tl::ascend::copy_gm_to_ub<float, 128, 128, 128, 128>(ubIn_[0], INPUT[0]);
  }
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
  {
    auto ubIn_ = ubIn.Get<float>();
    auto ubOut_ = ubOut.Get<half>();

    tl::ascend::cast<float, half, 128 * 128>(ubIn_[0], ubOut_[0]);

  }
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
  {
    auto ubOut_ = ubOut.Get<half>();
    tl::ascend::copy_ub_to_gm<half, 128, 128, 128, 128>(OUTPUT[0], ubOut_[0]);
  }
}

extern "C" void call(uint8_t* INPUT_handle, uint8_t* OUTPUT_handle, aclrtStream stream) {
  main_kernel<<<1, nullptr, stream>>>(INPUT_handle, OUTPUT_handle);
}