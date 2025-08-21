#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using namespace AscendC;

CATLASS_GLOBAL
void main_kernel(GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR V_handle,  GM_ADDR Output_handle, uint64_t fftsAddr) {
  AscendC::SetSyncBaseAddr(fftsAddr);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> Q;
  Q.SetGlobalBuffer((__gm__ half*)Q_handle);
  AscendC::GlobalTensor<half> K;
  K.SetGlobalBuffer((__gm__ half*)K_handle);
  AscendC::GlobalTensor<half> V;
  V.SetGlobalBuffer((__gm__ half*)V_handle);
  AscendC::GlobalTensor<float> Output;
  Output.SetGlobalBuffer((__gm__ float*)Output_handle);

  auto cid = AscendC::GetBlockIdx();
  AscendC::TBuf<AscendC::TPosition::A1> Q_L1;
  pipe.InitBuffer(Q_L1, 16384 * 2);
  AscendC::TBuf<AscendC::TPosition::A1> K_L1;
  pipe.InitBuffer(K_L1, 16384 * 2);
  AscendC::TBuf<AscendC::TPosition::A1> V_L1;
  pipe.InitBuffer(V_L1, 16384 * 2);

  AscendC::TBuf<AscendC::TPosition::VECCALC> acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum, acc_o;
  pipe.InitBuffer(acc_s, 16384 * 4);
  pipe.InitBuffer(scores_max, 128 * 4);
  pipe.InitBuffer(scores_max_prev, 128 * 4);
  pipe.InitBuffer(scores_scale, 128 * 4);
  pipe.InitBuffer(scores_sum, 128 * 4);
  pipe.InitBuffer(logsum, 128 * 4);
  pipe.InitBuffer(acc_o, 16384 * 4);

  AscendC::TBuf<AscendC::TPosition::VECCALC> acc_s_cast;
  pipe.InitBuffer(acc_s_cast, 16384 * 2);

  AscendC::TBuf<AscendC::TPosition::VECCALC> tmp;
  pipe.InitBuffer(tmp, 64 * 128 * 4 * 3);

  pipe.Destroy();

  if ASCEND_IS_AIC {
    auto Q_L1_ = Q_L1.Get<half>();
    tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 16384, 128, 128, 128>(Q_L1_[0], Q[cid * 128 * 128]);
    for (int k = 0; k < 128; ++k) {
      {
        auto K_L1_ = K_L1.Get<half>();
        tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 16384, 128, 128, 128>(K_L1_[0], K[k * 128 * 128]);
      }
      // {
      //   auto Q_L1_ = Q_L1.Get<half>();
      //   auto K_L1_ = K_L1.Get<half>();
      //   auto acc_s_ = acc_s.Get<float>();
      //   tl::ascend::gemm_ccu<half, float, 128, 128, 128, true, false, true>(Q_L1_[0], K_L1_[0], acc_s_[0]);
      // }
      AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(0);
      
      {
        auto V_L1_ = V_L1.Get<half>();
        tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 16384, 128, 128, 128>(V_L1_[0], V[k * 128 * 128]);
      }
      AscendC::CrossCoreWaitFlag(1);
      auto acc_s_cast_l1_ = K_L1.Get<half>();
      auto acc_s_cast_ = acc_s_cast.Get<half>();
      tl::ascend::copy_ub_to_l1<half, 128, 128>(acc_s_cast_l1_[0], acc_s_cast_[0]);

      // {
      //   auto acc_s_cast_l1_ = K_L1.Get<half>();
      //   auto V_L1_ = V_L1.Get<half>();
      //   auto acc_o_ = acc_o.Get<float>();
      //   tl::ascend::gemm_ccu<half, float, 128, 128, 128>(acc_s_cast_l1_[0], V_L1_[0], acc_o_[0]);
      // }
      AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(2);
    }
  }
  if ASCEND_IS_AIV {
    {
        auto logsum_ = logsum.Get<float>();
        tl::ascend::fill<float, 128>(logsum_[0], 0);
    }

    {
        auto scores_max_ = scores_max.Get<float>();
        tl::ascend::fill<float, 128>(scores_max_[0], -1.0f / 0.0f);

        auto acc_o_ = acc_o.Get<float>();
        tl::ascend::fill<float, 16384>(acc_o_[0], 0);
    }

    for (int k = 0; k < 128; ++k) {
        {
            auto scores_max_ = scores_max.Get<float>();
            auto scores_max_prev_ = scores_max_prev.Get<float>();

            AscendC::DataCopy(scores_max_prev_[0], scores_max_[0], 128);
        }

        {
            AscendC::CrossCoreWaitFlag(0);
            auto acc_s_ = acc_s.Get<float>();
            auto scores_max_ = scores_max.Get<float>();
            auto tmp_ = tmp.Get<uint8_t>();
            tl::ascend::reduce_max<float, 64, 128, Pattern::Reduce::AR>(acc_s_[0], scores_max_[0], tmp_[0]);

            tl::ascend::reduce_max<float, 64, 128, Pattern::Reduce::AR>(acc_s_[64 * 128], scores_max_[64 * 128], tmp_[0]);
        }

        {
            auto scores_max_prev_ = scores_max_prev.Get<float>();
            auto scores_max_ = scores_max.Get<float>();
            auto scores_scale_ = scores_scale.Get<float>();
            AscendC::Muls(scores_max_prev_[0], scores_max_prev_[0], 1.275174e-01f, 128);
            AscendC::Muls(scores_max_[0], scores_max_[0], 1.275174e-01f, 128);
            
            AscendC::Sub(scores_scale_[0], scores_max_prev_[0], scores_max_[0], 128);
            AscendC::Exp(scores_scale_[0], scores_scale_[0], 128);
            
            auto acc_s_ = acc_s.Get<float>();
            AscendC::Muls(acc_s_[0], acc_s_[0], 1.275174e-01f, 16384);
            AscendC::Muls(scores_max_[0], scores_max_[0], 1.275174e-01f, 16384);

            for (int i = 0; i < 128; ++i) {
                AscendC::Sub(acc_s_[i * 128], acc_s_[i * 128], scores_max_[0], 128);
            }
            AscendC::Exp(acc_s_[0], acc_s_[0], 16384);
            auto tmp_ = tmp.Get<uint8_t>();
            auto scores_sum_ = scores_sum.Get<float>();
            tl::ascend::reduce_sum<float, 64, 128, Pattern::Reduce::AR>(acc_s_[0], scores_sum_[0], tmp_[0]);
            tl::ascend::reduce_sum<float, 64, 128, Pattern::Reduce::AR>(acc_s_[64 * 128], scores_sum_[64 * 128], tmp_[0]);

            auto logsum_ = logsum.Get<float>();
            AscendC::Mul(logsum_[0], logsum_[0], scores_scale_[0], 128);
            AscendC::Add(logsum_[0], logsum_[0], scores_sum_[0], 128);

            auto acc_s_cast_ = acc_s_cast.Get<half>();

            tl::ascend::cast<float, half, 16384>(acc_s_[0], acc_s_cast_[0]);
            AscendC::CrossCoreSetFlag<2, PIPE_V>(1);
        }
        AscendC::CrossCoreWaitFlag(2);
        {
            for (int i = 0; i < 128; ++i) {
                auto acc_o_ = acc_o.Get<float>();
                auto scores_scale_ = scores_scale.Get<float>();

                AscendC::Mul(acc_o_[i * 128], acc_o_[i * 128], scores_scale_[0], 128);
            }
        }

    }

    for (int i = 0; i < 128; ++i) {
        auto acc_o_ = acc_o.Get<float>();
        auto logsum_ = logsum.Get<float>();
        AscendC::Div(acc_o_[i * 128], acc_o_[i * 128], logsum_[0], 128);
    }
    {
        auto acc_o_ = acc_o.Get<float>();
        tl::ascend::copy_ub_to_gm<float, 128, 128, 16384, 128>(Output[cid * 128 * 128], acc_o_[0]);
    }
  }
}

// cann-ops: MatMulImpl

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, uint8_t* D_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel<<<128, nullptr, stream>>>(A_handle, B_handle, C_handle, D_handle, fftsAddr);
}