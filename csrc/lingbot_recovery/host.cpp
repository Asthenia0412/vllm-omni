// SPDX-License-Identifier: Apache-2.0
// Reconstructed host entrypoints. Public torch-npu ops retain framework queue
// and allocator semantics. No historical binary/numerical admission is implied.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "copy_spec.h"

extern "C" int lingbot_recovery_copy_launch(void*,void*,void*,void*,void*,CopySpec,CopySpec);
extern "C" int lingbot_recovery_recv_launch(void*,void*,void*,void*,void*,void*,void*,void*,int64_t,int64_t,int64_t);
extern "C" int lingbot_recovery_send_launch(void*,void*,void*,void*,void*);

namespace {
using at::Tensor;
using I = std::vector<int64_t>;
namespace npu = at_npu::native::custom_ops;

void retain_on_stream(const std::vector<Tensor>& owners, c10_npu::NPUStream stream) {
  for (const auto& t : owners) {
    TORCH_CHECK(t.device().index() == stream.device_index(), "Wrong current device/stream");
    c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
  }
}

int64_t linked_copy_address() {
  return reinterpret_cast<int64_t>(&lingbot_recovery_copy_launch);
}

const torch_npu::NPUStorageDesc& descriptor(const Tensor& t) {
  TORCH_CHECK(t.defined() && t.device().type() == c10::DeviceType::PrivateUse1,
              "LingBot recovery requires an NPU tensor");
  // This wheel exports neither NPUStorageImpl RTTI nor NPUBridge. Match its
  // installed storage ABI only after checking the registered NPU device type.
  // The build/loader pins torch-npu and this header; no CPU storage is cast.
  auto* storage=static_cast<torch_npu::NPUStorageImpl*>(t.storage().unsafeGetStorageImpl());
  TORCH_CHECK(int64_t(storage->npu_desc_.npu_format_)==npu::get_npu_format(t),
              "Storage descriptor differs from public framework format");
  return storage->npu_desc_;
}

void dense_storage(const Tensor& t) {
  const auto& d = descriptor(t);
  TORCH_CHECK((d.npu_format_ == ACL_FORMAT_ND || d.npu_format_ == ACL_FORMAT_NCHW
               || d.npu_format_ == ACL_FORMAT_NCDHW) && d.origin_format_ == d.npu_format_
               && d.storage_sizes_ == d.base_sizes_,
              "Only unblocked ND/NCHW/NCDHW storage is supported; never retag storage");
  int64_t base_stride=1;
  for(int64_t dim=d.base_sizes_.size()-1; dim>=0; --dim) {
    TORCH_CHECK(d.base_strides_[dim]==base_stride,"Non-dense physical base");
    base_stride*=d.base_sizes_[dim];
  }
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.numel() > 0,
              "Nonempty BF16 tensor required");
  int64_t high = t.storage_offset();
  TORCH_CHECK(high >= 0, "Negative storage offset");
  for (int64_t i = 0; i < t.dim(); ++i) {
    TORCH_CHECK(t.stride(i) >= 0, "Negative stride");
    TORCH_CHECK(t.size(i) <= INT64_MAX / std::max<int64_t>(1, t.stride(i)), "Extent overflow");
    const auto delta = (t.size(i) - 1) * t.stride(i);
    TORCH_CHECK(high <= INT64_MAX - delta, "Extent overflow");
    high += delta;
  }
  TORCH_CHECK(high < static_cast<int64_t>(t.storage().nbytes() / t.element_size()),
              "View exceeds actual allocation");
}

auto describe(const Tensor& t) {
  const auto& d = descriptor(t);
  return std::make_tuple(int64_t(d.npu_format_), int64_t(d.origin_format_),
      I(d.base_sizes_.begin(), d.base_sizes_.end()),
      I(d.base_strides_.begin(), d.base_strides_.end()),
      I(d.storage_sizes_.begin(), d.storage_sizes_.end()),
      t.sizes().vec(), t.strides().vec(), t.storage_offset(), int64_t(t.storage().nbytes()),
      int64_t(reinterpret_cast<uintptr_t>(t.storage().data_ptr().get())),
      int64_t(reinterpret_cast<uintptr_t>(t.const_data_ptr())));
}

Tensor qk_send(const Tensor& raw, const Tensor& qw, const Tensor& kw,
               const Tensor& cosine, const Tensor& sine) {
  for (const auto& t : {raw, qw, kw, cosine, sine}) {
    dense_storage(t);
    TORCH_CHECK(t.device() == raw.device(), "Mixed NPU devices");
  }
  TORCH_CHECK(raw.sizes() == at::IntArrayRef({1,585,15360}) && raw.is_contiguous(),
              "SP8 raw QKV must be [1,585,15360]");
  TORCH_CHECK(qw.sizes() == at::IntArrayRef({5120}) && kw.sizes() == qw.sizes(),
              "RMSNorm must reduce all 5120 features");
  TORCH_CHECK(cosine.sizes() == at::IntArrayRef({1,585,1,128}) && sine.sizes() == cosine.sizes(),
              "Expanded interleaved rotary coefficients required");
  auto parts = raw.split(5120, -1);
  auto q = std::get<0>(npu::npu_rms_norm(parts[0], qw, 1e-6)).view({1,585,40,128});
  auto k = std::get<0>(npu::npu_rms_norm(parts[1], kw, 1e-6)).view({1,585,40,128});
  q = npu::npu_rotary_mul(q, cosine, sine, "interleave");
  k = npu::npu_rotary_mul(k, cosine, sine, "interleave");
  auto v = parts[2].view({1,585,40,128});
  auto send = npu::empty_with_format({8,585,3,1,5,128},at::kBFloat16,{},raw.device(),{},2);
  auto stream = c10_npu::getCurrentNPUStream(raw.device().index());
  std::vector<Tensor> owners{raw,q,k,v,send};
  retain_on_stream(owners,stream);
  at_npu::native::OpCommand::RunOpApiV2("lingbot_recovery_send", [owners,stream]() -> int {
    return lingbot_recovery_send_launch(stream.stream(false),owners[1].data_ptr(),owners[2].data_ptr(),
                                 owners[3].data_ptr(),owners[4].data_ptr());
  });
  return send;
}

CopySpec copy_spec(const Tensor& dst, const Tensor& src) {
  dense_storage(dst); dense_storage(src);
  TORCH_CHECK(dst.device()==src.device() && dst.sizes()==src.sizes(), "Copy descriptor mismatch");
  TORCH_CHECK((src.dim()==4 || src.dim()==5) && src.stride(-1)==1 && dst.stride(-1)==1,
              "Only 4D/5D row-contiguous copies supported");
  // Positive non-overlapping row-major subviews. Holes between rows are allowed.
  for (auto t : {src,dst}) {
    int64_t span=t.size(-1);
    for (int64_t d=t.dim()-2; d>=0; --d) {
      TORCH_CHECK(t.stride(d)>=span, "Overlapping or permuted strided copy");
      span+=(t.size(d)-1)*t.stride(d);
    }
  }
  CopySpec spec{}; spec.rows=1; spec.width=src.size(-1);
  int shift=5-src.dim();
  for (int d=0;d<5;++d) {
    spec.sizes[d]=d<shift?1:src.size(d-shift);
    spec.src_strides[d]=d<shift?0:src.stride(d-shift);
    spec.dst_strides[d]=d<shift?0:dst.stride(d-shift);
    if(d<4) spec.rows*=spec.sizes[d];
  }
  return spec;
}

I rectangle(const Tensor& base,const Tensor& view) {
  TORCH_CHECK(base.is_contiguous() && base.dim()==view.dim() && base.strides()==view.strides()
              && base.storage().unsafeGetStorageImpl()==view.storage().unsafeGetStorageImpl(),
              "Destination must be a rectangular view of the complete base");
  int64_t offset=view.storage_offset()-base.storage_offset();
  TORCH_CHECK(offset>=0,"Destination starts before base");
  I result;
  for(int d=0;d<base.dim();++d) {
    int64_t start=offset/base.stride(d); offset%=base.stride(d);
    TORCH_CHECK(start+view.size(d)<=base.size(d),"Destination rectangle exceeds base");
    result.push_back(start);
  }
  TORCH_CHECK(offset==0,"Misaligned rectangle");
  return result;
}

void copy_pair(const Tensor& base, Tensor da,const Tensor& sa,Tensor db,const Tensor& sb) {
  dense_storage(base);
  auto a=copy_spec(da,sa),b=copy_spec(db,sb);
  for(auto t:{da,sa,db,sb}) TORCH_CHECK(t.device()==base.device(),"Mixed devices");
  auto ra=rectangle(base,da),rb=rectangle(base,db);
  bool disjoint=false;
  for(int d=0;d<base.dim();++d)
    disjoint|=ra[d]+da.size(d)<=rb[d] || rb[d]+db.size(d)<=ra[d];
  TORCH_CHECK(disjoint,"Destination rectangles overlap");
  for(auto src:{sa,sb}) TORCH_CHECK(src.storage().unsafeGetStorageImpl()!=base.storage().unsafeGetStorageImpl(),
                                   "Source aliases writable arena");
  auto stream=c10_npu::getCurrentNPUStream(base.device().index());
  std::vector<Tensor> owners{base,da,sa,db,sb}; retain_on_stream(owners,stream);
  at_npu::native::OpCommand::RunOpApiV2("lingbot_recovery_copy_pair",[owners,stream,a,b]()->int {
    return lingbot_recovery_copy_launch(stream.stream(false),owners[1].data_ptr(),owners[2].data_ptr(),
                                 owners[3].data_ptr(),owners[4].data_ptr(),a,b);
  });
}

Tensor ingress(const Tensor& q,const Tensor& k,const Tensor& v,Tensor kp,Tensor vp,
               Tensor ks,Tensor vs,const Tensor& slots,int64_t first) {
  for(auto t:{q,k,v,kp,vp,ks,vs}) {
    dense_storage(t); TORCH_CHECK(t.device()==q.device(),"Mixed devices");
    TORCH_CHECK(t.dim()==3 && t.size(1)==5 && t.size(2)==128,"SP8 H5/D128 required");
  }
  for(int kind=0;kind<3;++kind) {
    const auto t=std::vector<Tensor>{q,k,v}[kind];
    TORCH_CHECK(t.sizes()==at::IntArrayRef({4680,5,128}) && t.strides()==at::IntArrayRef({1920,128,1})
        && t.storage().unsafeGetStorageImpl()==q.storage().unsafeGetStorageImpl()
        && t.storage_offset()==q.storage_offset()+kind*640,"Unexpected receive aliases/order");
  }
  TORCH_CHECK(kp.is_contiguous() && vp.is_contiguous() && ks.is_contiguous() && vs.is_contiguous()
      && kp.sizes()==vp.sizes() && ks.sizes()==vs.sizes(),"Invalid pool/staging capacities");
  TORCH_CHECK(first>=0 && first<=ks.size(0)/1560 && first*1560+4680<=ks.size(0),"Staging overflow");
  const auto& sd=descriptor(slots);
  TORCH_CHECK(sd.npu_format_==ACL_FORMAT_ND && slots.device()==q.device() && slots.scalar_type()==at::kLong
      && slots.sizes()==at::IntArrayRef({4680}) && slots.is_contiguous(),"Invalid slot descriptor");
  std::vector<Tensor> buffers{kp,vp,ks,vs};
  for(size_t i=0;i<buffers.size();++i) {
    TORCH_CHECK(buffers[i].storage().unsafeGetStorageImpl()!=q.storage().unsafeGetStorageImpl(),"Receive aliases output");
    TORCH_CHECK(buffers[i].storage().unsafeGetStorageImpl()!=slots.storage().unsafeGetStorageImpl(),"Slots alias output");
    for(size_t j=0;j<i;++j) TORCH_CHECK(buffers[i].storage().unsafeGetStorageImpl()!=buffers[j].storage().unsafeGetStorageImpl(),"Output owners alias");
  }
  auto result=npu::empty_with_format({4680,5,128},at::kBFloat16,{},q.device(),{},2);
  auto stream=c10_npu::getCurrentNPUStream(q.device().index());
  std::vector<Tensor> owners{q,k,v,kp,vp,ks,vs,slots,result}; retain_on_stream(owners,stream);
  at_npu::native::OpCommand::RunOpApiV2("lingbot_recovery_recv",[owners,stream,first]()->int {
    return lingbot_recovery_recv_launch(stream.stream(false),owners[0].data_ptr(),owners[8].data_ptr(),
      owners[3].data_ptr(),owners[4].data_ptr(),owners[5].data_ptr(),owners[6].data_ptr(),
      owners[7].data_ptr(),owners[3].size(0),owners[5].size(0),first*1560);
  });
  return result;
}

Tensor cross(const Tensor& q, const Tensor& k, const Tensor& v, double scale) {
  for (const auto& t : {q,k,v}) {
    dense_storage(t);
    TORCH_CHECK(t.device() == q.device() && t.is_contiguous(), "Dense same-device cross inputs required");
    TORCH_CHECK(t.dim() == 4 && t.size(0) == 1 && t.size(2) == 40 && t.size(3) == 128,
                "Cross attention requires BSND H40/D128");
  }
  TORCH_CHECK(q.size(1) == 585 && k.sizes() == v.sizes() && k.size(1) <= 512,
              "Unsupported cross attention sequence");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "Invalid attention scale");
  return std::get<0>(npu::npu_fused_infer_attention_score(q,k,v,
      {},{},std::nullopt,std::nullopt,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},
      std::nullopt,{},{},{},40,scale,2147483647,2147483647,"BSND",40));
}
} // namespace

TORCH_LIBRARY(lingbot_recovery_v1, m) {
  m.def("linked_copy_address() -> int", &linked_copy_address);
  m.def("describe(Tensor t) -> (int, int, int[], int[], int[], int[], int[], int, int, int, int)", &describe);
  m.def("qk_send(Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine) -> Tensor", &qk_send);
  m.def("cross(Tensor q, Tensor k, Tensor v, float scale) -> Tensor", &cross);
  m.def("copy_pair(Tensor(a!) base, Tensor(a!) da, Tensor sa, Tensor(a!) db, Tensor sb) -> ()", &copy_pair);
  m.def("ingress(Tensor q, Tensor k, Tensor v, Tensor(a!) kp, Tensor(b!) vp, Tensor(c!) ks, Tensor(d!) vs, Tensor slots, int first) -> Tensor", &ingress);
}
