// SPDX-License-Identifier: Apache-2.0
// Reconstructed host entrypoints. Public torch-npu ops retain framework queue
// and allocator semantics. No historical binary/numerical admission is implied.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "copy_spec.h"

extern "C" int lingbot_recovery_copy_launch(void*,void*,void*,void*,void*,CopySpec,CopySpec);
extern "C" int lingbot_recovery_recv_launch(void*,void*,void*,void*,void*,void*,void*,void*,int64_t,int64_t,int64_t);
extern "C" int lingbot_recovery_send_launch(void*,void*,void*,void*,void*);

#include <atomic>
#include <mutex>
#include <memory>
#include <array>
#include <dlfcn.h>
#include <c10/core/InferenceMode.h>
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "aclnnop/aclnn_rotary_position_embedding_v2.h"

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


// Isolated candidate: NOT wired into the model and NOT device admitted.
struct NativeStep {
  uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  at::Tensor workspace;
};
struct CallState {
  std::vector<Tensor> owners;
  std::vector<aclTensor*> metadata;
  std::array<NativeStep,4> stages;
  std::atomic<bool> submitted{false};
  ~CallState() {
    // Successful non-repeatable native second-stage calls consume executors.
    // Failure states are quarantined without destruction until process exit.
    for (auto* desc : metadata) if (desc) aclDestroyTensor(desc);
  }
};
std::mutex quarantine_mutex;
std::vector<std::shared_ptr<CallState>> quarantine;
std::atomic<bool> poisoned{false};
void poison(const std::shared_ptr<CallState>& state) {
  poisoned.store(true);
  std::lock_guard<std::mutex> guard(quarantine_mutex);
  quarantine.push_back(state);
}
aclTensor* acl_view(const Tensor& t, CallState& state) {
  const auto& d=descriptor(t);
  aclDataType dtype=t.scalar_type()==at::kBFloat16 ? ACL_BF16 : ACL_FLOAT;
  TORCH_CHECK(t.scalar_type()==at::kBFloat16 || t.scalar_type()==at::kFloat,
              "Only BF16 arithmetic and FP32 rstd metadata allowed");
  // Allocation base plus explicit view offset, never tensor data_ptr+offset twice.
  auto* desc=aclCreateTensor(t.sizes().data(),t.dim(),dtype,t.strides().data(),
      t.storage_offset(),d.npu_format_,d.storage_sizes_.data(),d.storage_sizes_.size(),
      t.storage().data_ptr().get());
  TORCH_CHECK(desc,"aclCreateTensor failed");
  state.metadata.push_back(desc);
  return desc;
}
void workspace(NativeStep& step, CallState& state, c10::Device device) {
  if (!step.bytes) return;
  TORCH_CHECK(step.bytes <= INT64_MAX,"workspace overflow");
  step.workspace=npu::empty_with_format({int64_t(step.bytes)},at::kByte,{},device,{},2);
  state.owners.push_back(step.workspace);
}
std::vector<std::string> dependencies() {
  std::vector<std::string> paths;
  for (auto* f : {reinterpret_cast<void*>(&aclnnRmsNormGetWorkspaceSize),
                 reinterpret_cast<void*>(&aclnnRmsNorm),
                 reinterpret_cast<void*>(&aclnnRotaryPositionEmbeddingV2GetWorkspaceSize),
                 reinterpret_cast<void*>(&aclnnRotaryPositionEmbeddingV2),
                 reinterpret_cast<void*>(&aclCreateTensor),
                 reinterpret_cast<void*>(&aclDestroyTensor),
                 reinterpret_cast<void*>(&lingbot_recovery_send_launch)}) {
    Dl_info info{};
    TORCH_CHECK(dladdr(f,&info) && info.dli_fname,"unbound SDK symbol");
    paths.emplace_back(info.dli_fname);
  }
  return paths;
}
std::tuple<Tensor,Tensor> single_queue_impl(const Tensor& raw,const Tensor& qw,const Tensor& kw,
                                     const Tensor& cosine,const Tensor& sine, const Tensor* test_arena) {
  TORCH_CHECK(!poisoned.load(),"QK prototype poisoned; restart entire worker group");
  TORCH_CHECK(c10::InferenceMode::is_enabled() && !at::GradMode::is_enabled(),
              "inference-only isolated prototype");
  // RunOpApiV2 submits on the thread's current device, not raw.device().
  // Establish this before capture/stream queries, native planning or allocation.
  TORCH_CHECK(raw.defined() && raw.device().type()==c10::DeviceType::PrivateUse1,
              "QK single-queue requires a defined NPU raw tensor");
  TORCH_CHECK(c10_npu::current_device()==raw.device().index(),
              "QK single-queue current device differs from raw device before allocation");
  TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx()==c10_npu::CaptureStatus::None,
              "No graph capture");
  auto stream=c10_npu::getCurrentNPUStream(raw.device().index());
  TORCH_CHECK(stream==c10_npu::getDefaultNPUStream(raw.device().index()),"default stream only");
  for(const auto& t:{raw,qw,kw,cosine,sine}) {
    dense_storage(t);
    TORCH_CHECK(t.device()==raw.device(),"mixed devices");
  }
  TORCH_CHECK(raw.sizes()==at::IntArrayRef({1,585,15360}) && raw.is_contiguous(),"raw ABI");
  TORCH_CHECK(qw.sizes()==at::IntArrayRef({5120}) && kw.sizes()==qw.sizes(),"gamma ABI");
  TORCH_CHECK(cosine.sizes()==at::IntArrayRef({1,585,1,128}) && sine.sizes()==cosine.sizes(),"coefficient ABI");
  if(test_arena) {
    dense_storage(*test_arena);
    TORCH_CHECK(test_arena->device()==raw.device() && test_arena->sizes()==at::IntArrayRef({8986112})
        && test_arena->is_contiguous() && test_arena->storage_offset()==0
        && test_arena->storage().nbytes()==17972224,"test arena must be complete exact owner");
    for(const auto& t:{raw,qw,kw,cosine,sine})at::assert_no_overlap(*test_arena,t);
  }
  auto state=std::make_shared<CallState>();
  state->owners={raw,qw,kw,cosine,sine};
  try {
    auto allocate=[&](at::IntArrayRef shape,at::ScalarType dtype) {
      auto t=npu::empty_with_format(shape,dtype,{},raw.device(),{},2);
      state->owners.push_back(t);return t;
    };
    auto qnorm=allocate({1,585,5120},at::kBFloat16);
    auto knorm=allocate({1,585,5120},at::kBFloat16);
    auto qrstd=allocate({1,585,1},at::kFloat);
    auto krstd=allocate({1,585,1},at::kFloat);
    auto qrope=allocate({1,585,40,128},at::kBFloat16);
    auto krope=allocate({1,585,40,128},at::kBFloat16);
    auto owner=test_arena ? *test_arena : allocate({8986112},at::kBFloat16);
    if(test_arena)state->owners.push_back(owner);
    auto send=owner.as_strided({8,585,3,1,5,128},{1123200,1920,640,640,128,1},256);
    auto parts=raw.split(5120,-1);
    auto v=parts[2].view({1,585,40,128});
    state->owners.insert(state->owners.end(),{send,v});
    auto* qinput=acl_view(parts[0],*state);auto* kinput=acl_view(parts[1],*state);
    auto* qgamma=acl_view(qw,*state);auto* kgamma=acl_view(kw,*state);
    auto* qn=acl_view(qnorm,*state);auto* kn=acl_view(knorm,*state);
    auto* qs=acl_view(qrstd,*state);auto* ks=acl_view(krstd,*state);
    auto& a=state->stages;
    TORCH_CHECK(aclnnRmsNormGetWorkspaceSize(qinput,qgamma,1e-6,qn,qs,&a[0].bytes,&a[0].executor)==0,"Q RMS prepare");
    TORCH_CHECK(aclnnRmsNormGetWorkspaceSize(kinput,kgamma,1e-6,kn,ks,&a[1].bytes,&a[1].executor)==0,"K RMS prepare");
    auto* q4=acl_view(qnorm.view({1,585,40,128}),*state);
    auto* k4=acl_view(knorm.view({1,585,40,128}),*state);
    auto* cos=acl_view(cosine,*state);auto* sin=acl_view(sine,*state);
    auto* qo=acl_view(qrope,*state);auto* ko=acl_view(krope,*state);
    // Exact installed op-plugin gitlink8b9c8534 OpUtils.cpp maps interleave=1.
    TORCH_CHECK(aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(q4,cos,sin,1,nullptr,qo,&a[2].bytes,&a[2].executor)==0,"Q RoPE prepare");
    TORCH_CHECK(aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(k4,cos,sin,1,nullptr,ko,&a[3].bytes,&a[3].executor)==0,"K RoPE prepare");
    for(auto& step:a)workspace(step,*state,raw.device());
    retain_on_stream(state->owners,stream);
    // Captured tensor aliases keep native intermediate buffers alive. No
    // framework operator or allocator call occurs inside this queued callback.
    at_npu::native::OpCommand::RunOpApiV2("lingbot_qk_single_queue_guard_v2",
      [state,stream,qrope,krope,v,send]()->int {
        state->submitted.store(true);
        try {
          auto aclstream=stream.stream(false);
          for(size_t i=0;i<state->stages.size();++i) {
            auto& step=state->stages[i];
            auto* exec=step.executor;
            // Non-repeatable execution owns destruction, including its normal
            // return. Never call aclDestroyAclOpExecutor afterward.
            step.executor=nullptr;
            void* ptr=step.bytes ? step.workspace.data_ptr() : nullptr;
            int result=i<2 ? aclnnRmsNorm(ptr,step.bytes,exec,aclstream)
                          : aclnnRotaryPositionEmbeddingV2(ptr,step.bytes,exec,aclstream);
            if(result){poison(state);return result;}
          }
          int result=lingbot_recovery_send_launch(aclstream,qrope.data_ptr(),krope.data_ptr(),v.data_ptr(),send.data_ptr());
          if(result)poison(state);
          return result;
        } catch(...) {poison(state);throw;}
      });
    return {send,owner};
  } catch(...) {
    // Covers failed executor preparation/allocation/enqueue, including opaque
    // non-repeatable executors for which public API gives no cancellation path.
    // Fail the request and retire workers; keep all owners until then.
    poison(state);throw;
  }
}
std::tuple<Tensor,Tensor> single_queue(const Tensor& raw,const Tensor& qw,const Tensor& kw,
                                     const Tensor& cosine,const Tensor& sine) {
  return single_queue_impl(raw,qw,kw,cosine,sine,nullptr);
}
std::tuple<Tensor,Tensor> single_queue_test_arena(const Tensor& raw,const Tensor& qw,const Tensor& kw,
                                     const Tensor& cosine,const Tensor& sine,const Tensor& arena) {
  return single_queue_impl(raw,qw,kw,cosine,sine,&arena);
}
} // namespace
TORCH_LIBRARY(lingbot_qk_single_queue_guard_v2,m) {
  m.def("run(Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine) -> (Tensor(a), Tensor(a))",&single_queue);
  m.def("dependencies() -> str[]",&dependencies);
  m.def("run_with_test_arena(Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine, Tensor(a!) arena) -> (Tensor(a!), Tensor(a!))",&single_queue_test_arena);
}
