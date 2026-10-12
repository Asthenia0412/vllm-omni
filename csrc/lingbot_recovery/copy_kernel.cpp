// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "copy_spec.h"
using namespace AscendC;

// Bit transport only: interpreting BF16 as uint16 preserves every payload bit.
// One 32 KiB UB allocation, no Cube/L1/L0 usage. Each core owns whole rows.
class CopyRows {
 public:
  __aicore__ inline void Init() { pipe.InitBuffer(buffer, 32768); }
  __aicore__ inline void Pair(GM_ADDR dst, GM_ADDR src, const CopySpec& spec) {
    GlobalTensor<uint16_t> in, out;
    in.SetGlobalBuffer((__gm__ uint16_t*)src);
    out.SetGlobalBuffer((__gm__ uint16_t*)dst);
    LocalTensor<uint16_t> local = buffer.Get<uint16_t>();
    for (int64_t row = GetBlockIdx(); row < spec.rows; row += GetBlockNum()) {
      int64_t index = row, si = 0, di = 0;
      for (int d = 3; d >= 0; --d) {
        const int64_t coord = index % spec.sizes[d];
        index /= spec.sizes[d];
        si += coord * spec.src_strides[d]; di += coord * spec.dst_strides[d];
      }
      for (int64_t col = 0; col < spec.width; col += 16384) {
        const uint32_t count = static_cast<uint32_t>(spec.width-col < 16384 ? spec.width-col : 16384);
        DataCopyExtParams params{1, count * 2, 0, 0, 0};
        DataCopyPad(local, in[si+col], params, DataCopyPadExtParams<uint16_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(out[di+col], local, params);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
  }
 private:
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
};

extern "C" __global__ __aicore__ void lingbot_recovery_copy_pair(
    GM_ADDR da, GM_ADDR sa, GM_ADDR db, GM_ADDR sb, CopySpec a, CopySpec b) {
  CopyRows op; op.Init(); op.Pair(da,sa,a); op.Pair(db,sb,b);
}

extern "C" int lingbot_recovery_copy_launch(void* stream, void* da, void* sa,
    void* db, void* sb, CopySpec a, CopySpec b) {
  return lingbot_recovery_copy_pair<<<48, nullptr, stream>>>((GM_ADDR)da,(GM_ADDR)sa,(GM_ADDR)db,(GM_ADDR)sb,&a,&b);
}

extern "C" __global__ __aicore__ void lingbot_recovery_recv(
    GM_ADDR recv, GM_ADDR query, GM_ADDR kp, GM_ADDR vp, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR slots, int64_t pool_rows, int64_t stage_rows, int64_t first_token) {
  TPipe pipe; TBuf<TPosition::VECCALC> buffer; pipe.InitBuffer(buffer, 1280);
  auto local = buffer.Get<uint16_t>();
  GlobalTensor<uint16_t> in, q, pk, pv, sk, sv;
  GlobalTensor<int64_t> ids;
  in.SetGlobalBuffer((__gm__ uint16_t*)recv); q.SetGlobalBuffer((__gm__ uint16_t*)query);
  pk.SetGlobalBuffer((__gm__ uint16_t*)kp); pv.SetGlobalBuffer((__gm__ uint16_t*)vp);
  sk.SetGlobalBuffer((__gm__ uint16_t*)ks); sv.SetGlobalBuffer((__gm__ uint16_t*)vs);
  ids.SetGlobalBuffer((__gm__ int64_t*)slots);
  for (int64_t token=GetBlockIdx(); token<4680; token+=GetBlockNum()) {
    const int64_t slot=ids.GetValue(token), stage=first_token+token;
    ASSERT(slot >= 0 && slot < pool_rows && stage >= 0 && stage < stage_rows);
    for (int64_t kind=0; kind<3; ++kind) {
      DataCopy(local, in[token*1920+kind*640], 640);
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      if (kind==0) { DataCopy(q[token*640], local, 640); }
      else if (kind==1) { DataCopy(pk[slot*640], local, 640); DataCopy(sk[stage*640], local, 640); }
      else { DataCopy(pv[slot*640], local, 640); DataCopy(sv[stage*640], local, 640); }
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }
}

extern "C" int lingbot_recovery_recv_launch(void* stream, void* recv, void* q,
    void* kp, void* vp, void* ks, void* vs, void* slots,
    int64_t pool_rows, int64_t stage_rows, int64_t first_token) {
  return lingbot_recovery_recv<<<48, nullptr, stream>>>((GM_ADDR)recv,(GM_ADDR)q,(GM_ADDR)kp,
      (GM_ADDR)vp,(GM_ADDR)ks,(GM_ADDR)vs,(GM_ADDR)slots,pool_rows,stage_rows,first_token);
}

extern "C" __global__ __aicore__ void lingbot_recovery_send(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR send) {
  TPipe pipe; TBuf<TPosition::VECCALC> buffer; pipe.InitBuffer(buffer, 1280);
  auto local = buffer.Get<uint16_t>();
  GlobalTensor<uint16_t> qs,ks,vs,out;
  qs.SetGlobalBuffer((__gm__ uint16_t*)q); ks.SetGlobalBuffer((__gm__ uint16_t*)k);
  vs.SetGlobalBuffer((__gm__ uint16_t*)v); out.SetGlobalBuffer((__gm__ uint16_t*)send);
  for (int64_t row=GetBlockIdx(); row<8*585; row+=GetBlockNum()) {
    const int64_t peer=row/585, token=row%585;
    for (int64_t kind=0; kind<3; ++kind) {
      if (kind==0) DataCopy(local,qs[token*5120+peer*640],640);
      else if (kind==1) DataCopy(local,ks[token*5120+peer*640],640);
      else DataCopy(local,vs[token*15360+peer*640],640);
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      DataCopy(out[row*1920+kind*640],local,640);
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }
}

extern "C" int lingbot_recovery_send_launch(void* stream, void* q, void* k, void* v, void* send) {
  return lingbot_recovery_send<<<48,nullptr,stream>>>((GM_ADDR)q,(GM_ADDR)k,(GM_ADDR)v,(GM_ADDR)send);
}
