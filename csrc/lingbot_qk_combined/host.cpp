// SPDX-License-Identifier: Apache-2.0
// Reconstructed host entrypoints. Public torch-npu ops retain framework queue
// and allocator semantics. No historical binary/numerical admission is implied.
#include <ATen/ATen.h>
#include <optional>
#include <ATen/MemoryOverlap.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" int lingbot_recovery_send_launch(void*,void*,void*,void*,void*);

#include <atomic>
#include <mutex>
#include <memory>
#include <dlfcn.h>

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

bool dense_admitted(const Tensor& t) {
  const auto& d = descriptor(t);
  if (!(d.npu_format_==ACL_FORMAT_ND || d.npu_format_==ACL_FORMAT_NCHW
        || d.npu_format_==ACL_FORMAT_NCDHW)) return false;
  TORCH_CHECK(d.origin_format_==d.npu_format_ && d.storage_sizes_==d.base_sizes_
      && d.base_sizes_.size()==d.base_strides_.size(), "Invalid QK physical metadata");
  int64_t extent=1;
  for (int64_t i=int64_t(d.base_sizes_.size())-1;i>=0;--i) {
    TORCH_CHECK(d.base_sizes_[i]>0 && d.base_strides_[i]==extent
        && extent<=INT64_MAX/d.base_sizes_[i], "Invalid QK physical extent/stride");
    extent*=d.base_sizes_[i];
  }
  const auto capacity=t.storage().nbytes()/t.element_size();
  TORCH_CHECK(uint64_t(extent)<=capacity,"QK physical base exceeds owner");
  int64_t high=t.storage_offset();
  TORCH_CHECK(high>=0,"Negative QK storage offset");
  for(int64_t i=0;i<t.dim();++i) {
    TORCH_CHECK(t.size(i)>=0 && t.stride(i)>=0,"Invalid QK view metadata");
    if(t.size(i)==0)return false;
    TORCH_CHECK(t.size(i)-1<=INT64_MAX/std::max<int64_t>(1,t.stride(i)),"QK extent overflow");
    const auto delta=(t.size(i)-1)*t.stride(i);
    TORCH_CHECK(high<=INT64_MAX-delta,"QK extent overflow");high+=delta;
  }
  TORCH_CHECK(uint64_t(high)<capacity,"QK view exceeds owner");
  return true;
}
void dense_storage(const Tensor& t) {
  TORCH_CHECK(t.scalar_type()==at::kBFloat16 && dense_admitted(t),"Unsupported QK storage");
}

std::atomic<bool> poisoned{false};

bool admitted(const Tensor& raw,const Tensor& qw,const Tensor& kw,
              const Tensor& cosine,const Tensor& sine,bool compatible) {
  TORCH_CHECK(!poisoned.load(),"QK route poisoned; restart all eight workers");
  if(!compatible || c10::GradMode::is_enabled() || !c10::InferenceMode::is_enabled())return false;
  for(const auto& t:{raw,qw,kw,cosine,sine}) {
    TORCH_CHECK(t.defined(),"Undefined QK input");
    if(t.device().type()!=c10::DeviceType::PrivateUse1 || t.scalar_type()!=at::kBFloat16)return false;
  }
  for(const auto& t:{qw,kw,cosine,sine})TORCH_CHECK(t.device()==raw.device(),"Mixed QK devices");
  TORCH_CHECK(c10_npu::current_device()==raw.device().index(),"QK current device differs from owner");
  auto stream=c10_npu::getCurrentNPUStream(raw.device().index());
  if(stream!=c10_npu::getDefaultNPUStream(raw.device().index()))return false;
  const auto capture=c10_npu::currentStreamCaptureStatusMayInitCtx();
  TORCH_CHECK(capture!=c10_npu::CaptureStatus::Invalidated,"QK capture invalidated");
  if(capture!=c10_npu::CaptureStatus::None)return false;
  for(const auto& t:{raw,qw,kw,cosine,sine})if(!dense_admitted(t))return false;
  return raw.sizes()==at::IntArrayRef({1,585,15360}) && raw.is_contiguous()
      && qw.sizes()==at::IntArrayRef({5120}) && kw.sizes()==qw.sizes()
      && cosine.sizes()==at::IntArrayRef({1,585,1,128}) && sine.sizes()==cosine.sizes();
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

// Preserve the installed public op-plugin's native workspace cache, ancillary
// policy handling and per-op queues. One host admission does not mean one queue.
struct Owners {std::vector<Tensor> tensors;};
std::mutex quarantine_mutex;
std::vector<std::shared_ptr<Owners>> quarantine;
void poison(const std::shared_ptr<Owners>& owners) {
  poisoned.store(true);
  std::lock_guard<std::mutex> guard(quarantine_mutex);quarantine.push_back(owners);
}
std::vector<std::string> dependencies() {
  Dl_info info{};
  TORCH_CHECK(dladdr(reinterpret_cast<void*>(&lingbot_recovery_send_launch),&info)
      && info.dli_fname,"Unbound verified send launcher");
  return {info.dli_fname};
}
std::tuple<Tensor,Tensor> native_impl(const Tensor& raw,const Tensor& qw,const Tensor& kw,
    const Tensor& cosine,const Tensor& sine,const Tensor* test_arena) {
  if(test_arena) {
    dense_storage(*test_arena);
    TORCH_CHECK(test_arena->device()==raw.device() && test_arena->sizes()==at::IntArrayRef({8986112})
        && test_arena->is_contiguous() && test_arena->storage_offset()==0
        && test_arena->storage().nbytes()==17972224,"Test arena must be complete exact owner");
    for(const auto& t:{raw,qw,kw,cosine,sine})at::assert_no_overlap(*test_arena,t);
  }
  auto owners=std::make_shared<Owners>();owners->tensors={raw,qw,kw,cosine,sine};
  try {
    auto parts=raw.split(5120,-1);
    // Same source operation order and explicit BF16 boundaries as qk_send.
    auto q=std::get<0>(npu::npu_rms_norm(parts[0],qw,1e-6)).view({1,585,40,128});
    owners->tensors.push_back(q);
    auto k=std::get<0>(npu::npu_rms_norm(parts[1],kw,1e-6)).view({1,585,40,128});
    owners->tensors.push_back(k);
    q=npu::npu_rotary_mul(q,cosine,sine,"interleave");owners->tensors.push_back(q);
    k=npu::npu_rotary_mul(k,cosine,sine,"interleave");owners->tensors.push_back(k);
    auto v=parts[2].view({1,585,40,128});
    auto owner=test_arena ? *test_arena : npu::empty_with_format({8986112},at::kBFloat16,{},raw.device(),{},2);
    auto send=owner.as_strided({8,585,3,1,5,128},{1123200,1920,640,640,128,1},256);
    owners->tensors.insert(owners->tensors.end(),{v,owner,send});
    auto stream=c10_npu::getCurrentNPUStream(raw.device().index());
    retain_on_stream(owners->tensors,stream);
    at_npu::native::OpCommand::RunOpApiV2("lingbot_qk_combined_send_v1",
      [owners,stream,q,k,v,send]()->int {
        try {
          int result=lingbot_recovery_send_launch(stream.stream(false),q.data_ptr(),k.data_ptr(),v.data_ptr(),send.data_ptr());
          if(result)poison(owners);return result;
        } catch(...) {poison(owners);throw;}
      });
    return {send,owner};
  } catch(...) {poison(owners);throw;}
}
void validate_lease(const Tensor& raw,const Tensor& metadata,const Tensor& send,
                    const Tensor& owner,int64_t raw_storage,int64_t raw_data,int64_t owner_data) {
  TORCH_CHECK(!poisoned.load(),"QK provider poisoned before HCCL");
  for(const auto& t:{raw,metadata,send,owner})dense_storage(t);
  TORCH_CHECK(c10_npu::current_device()==raw.device().index()
      && metadata.device()==raw.device() && send.device()==raw.device() && owner.device()==raw.device(),
      "QK lease device mismatch");
  TORCH_CHECK(metadata.sizes()==at::IntArrayRef({1,585,3,40,128})
      && metadata.strides()==at::IntArrayRef({8985600,15360,5120,128,1})
      && metadata.storage().unsafeGetStorageImpl()==raw.storage().unsafeGetStorageImpl()
      && metadata.storage_offset()==raw.storage_offset(),"QK metadata/raw association changed");
  TORCH_CHECK(reinterpret_cast<int64_t>(raw.storage().data_ptr().get())==raw_storage
      && reinterpret_cast<int64_t>(raw.data_ptr())==raw_data,"QK raw owner changed");
  const auto& d=descriptor(send);
  TORCH_CHECK(d.npu_format_==ACL_FORMAT_ND && d.origin_format_==ACL_FORMAT_ND
      && I(d.base_sizes_.begin(),d.base_sizes_.end())==I({8986112})
      && I(d.base_strides_.begin(),d.base_strides_.end())==I({1})
      && d.storage_sizes_==d.base_sizes_ && send.storage().nbytes()==17972224,
      "QK full padded owner changed");
  TORCH_CHECK(send.sizes()==at::IntArrayRef({8,585,3,1,5,128})
      && send.strides()==at::IntArrayRef({1123200,1920,640,640,128,1}) && send.storage_offset()==256
      && owner.sizes()==at::IntArrayRef({8986112}) && owner.is_contiguous() && owner.storage_offset()==0
      && send.storage().unsafeGetStorageImpl()==owner.storage().unsafeGetStorageImpl()
      && reinterpret_cast<int64_t>(owner.data_ptr())==owner_data
      && reinterpret_cast<int64_t>(send.data_ptr())==owner_data+512,"QK send owner/offset changed");
}
using OptionalOutputs=std::tuple<std::optional<Tensor>,std::optional<Tensor>>;
OptionalOutputs try_run(const Tensor& raw,const Tensor& qw,const Tensor& kw,
                        const Tensor& cosine,const Tensor& sine,bool compatible) {
  if(!admitted(raw,qw,kw,cosine,sine,compatible))return {std::nullopt,std::nullopt};
  auto [send,owner]=native_impl(raw,qw,kw,cosine,sine,nullptr);
  return {send,owner};
}
std::tuple<Tensor,Tensor> run_test(const Tensor& raw,const Tensor& qw,const Tensor& kw,
                                const Tensor& cosine,const Tensor& sine,const Tensor& arena) {
  TORCH_CHECK(admitted(raw,qw,kw,cosine,sine,true),"Test input unsupported");
  return native_impl(raw,qw,kw,cosine,sine,&arena);
}
} // namespace
TORCH_LIBRARY(lingbot_qk_combined_v1,m) {
  m.def("try_run(Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine, bool compatible) -> (Tensor(a)?, Tensor(a)?)",&try_run);
  m.def("dependencies() -> str[]",&dependencies);
  m.def("validate_lease(Tensor raw, Tensor metadata, Tensor send, Tensor owner, int raw_storage, int raw_data, int owner_data) -> ()",&validate_lease);
  m.def("describe(Tensor t) -> (int, int, int[], int[], int[], int[], int[], int, int, int, int)",&describe);
  m.def("run_with_test_arena(Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine, Tensor(a!) arena) -> (Tensor(a!), Tensor(a!))",&run_test);
}
