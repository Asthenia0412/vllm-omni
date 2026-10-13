// SPDX-License-Identifier: Apache-2.0
// Independent private-retention v3. No production registration or inherited qualification.
#include <ATen/ATen.h>
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
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "aclnnop/aclnn_rotary_position_embedding_v2.h"
#include <array>
#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" int lingbot_recovery_send_launch(void*, void*, void*, void*, void*);
namespace {
using at::Tensor;
using I = std::vector<int64_t>;
namespace npu = at_npu::native::custom_ops;
constexpr int64_t OWNER_ELEMENTS = 8986112;
constexpr int64_t OWNER_BYTES = 17972224;
int64_t tid() { return static_cast<int64_t>(syscall(SYS_gettid)); }

const torch_npu::NPUStorageDesc& descriptor(const Tensor& t) {
  TORCH_CHECK(t.defined() && t.device().type()==c10::DeviceType::PrivateUse1,
              "Defined NPU tensor required");
  auto* s=static_cast<torch_npu::NPUStorageImpl*>(t.storage().unsafeGetStorageImpl());
  TORCH_CHECK(int64_t(s->npu_desc_.npu_format_)==npu::get_npu_format(t),
              "Physical descriptor differs from framework format");
  return s->npu_desc_;
}
void dense(const Tensor& t, at::ScalarType dtype) {
  const auto& d=descriptor(t);
  TORCH_CHECK(t.scalar_type()==dtype && t.numel()>0, "Unexpected dtype/empty tensor");
  TORCH_CHECK((d.npu_format_==ACL_FORMAT_ND || d.npu_format_==ACL_FORMAT_NCHW ||
              d.npu_format_==ACL_FORMAT_NCDHW) && d.origin_format_==d.npu_format_ &&
              d.storage_sizes_==d.base_sizes_ && d.base_sizes_.size()==d.base_strides_.size(),
              "Unsupported physical owner; no format retagging");
  int64_t extent=1;
  for(int64_t i=int64_t(d.base_sizes_.size())-1;i>=0;--i) {
    TORCH_CHECK(d.base_sizes_[i]>0 && d.base_strides_[i]==extent &&
                extent<=INT64_MAX/d.base_sizes_[i], "Invalid physical extent/stride");
    extent*=d.base_sizes_[i];
  }
  TORCH_CHECK(uint64_t(extent)<=t.storage().nbytes()/t.element_size(), "Physical owner overflow");
  int64_t high=t.storage_offset();
  TORCH_CHECK(high>=0, "Negative view offset");
  for(int64_t i=0;i<t.dim();++i) {
    TORCH_CHECK(t.size(i)>0 && t.stride(i)>=0 &&
                t.size(i)-1<=INT64_MAX/std::max<int64_t>(1,t.stride(i)), "View extent overflow");
    auto delta=(t.size(i)-1)*t.stride(i);
    TORCH_CHECK(high<=INT64_MAX-delta, "View overflow"); high+=delta;
  }
  TORCH_CHECK(uint64_t(high)<t.storage().nbytes()/t.element_size(), "View outside owner");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.storage().data_ptr().get())%32==0 &&
              reinterpret_cast<uintptr_t>(t.const_data_ptr())%32==0, "32-byte alignment required");
}
void exact_arena(const Tensor& t) {
  dense(t,at::kBFloat16);const auto& d=descriptor(t);
  TORCH_CHECK(t.sizes()==at::IntArrayRef({OWNER_ELEMENTS}) && t.is_contiguous() &&
      t.storage_offset()==0 && t.storage().nbytes()==OWNER_BYTES &&
      d.npu_format_==ACL_FORMAT_ND && d.origin_format_==ACL_FORMAT_ND &&
      d.base_sizes_.size()==1 && d.base_sizes_[0]==OWNER_ELEMENTS &&
      d.storage_sizes_.size()==1 && d.storage_sizes_[0]==OWNER_ELEMENTS &&
      d.base_strides_.size()==1 && d.base_strides_[0]==1,
      "Exact padded ND owner/physical extent required");
}
// Complete non-address cache key; different layouts never reuse a plan.
I signature(const Tensor& t) {
  const auto& d=descriptor(t);
  I r{int64_t(t.scalar_type()),t.device().index(),int64_t(d.npu_format_),
      int64_t(d.origin_format_),t.storage_offset(),int64_t(t.storage().nbytes())};
  auto add=[&](const auto& values) {r.push_back(values.size());r.insert(r.end(),values.begin(),values.end());};
  add(d.base_sizes_);add(d.base_strides_);add(d.storage_sizes_);add(t.sizes());add(t.strides());
  return r;
}
aclrtContext context() {
  aclrtContext value=nullptr;
  TORCH_CHECK(aclrtGetCurrentContext(&value)==ACL_SUCCESS && value, "No current ACL context");
  return value;
}
c10_npu::NPUStream admit(const std::array<Tensor,5>& inputs) {
  const auto& raw=inputs[0];
  TORCH_CHECK(raw.defined() && raw.device().type()==c10::DeviceType::PrivateUse1, "NPU raw required");
  TORCH_CHECK(c10_npu::current_device()==raw.device().index(), "Current device differs from raw");
  TORCH_CHECK(c10::InferenceMode::is_enabled() && !at::GradMode::is_enabled(), "Inference only");
  TORCH_CHECK(!at::globalContext().deterministicAlgorithms(), "Prototype requires frozen nondeterministic policy");
  TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx()==c10_npu::CaptureStatus::None, "No graph capture");
  auto stream=c10_npu::getCurrentNPUStream(raw.device().index());
  TORCH_CHECK(stream==c10_npu::getDefaultNPUStream(raw.device().index()), "Default stream only");
  for(const auto& t:inputs) {dense(t,at::kBFloat16);TORCH_CHECK(t.device()==raw.device(), "Mixed devices");}
  TORCH_CHECK(raw.sizes()==at::IntArrayRef({1,585,15360}) && raw.is_contiguous(), "Raw ABI");
  TORCH_CHECK(inputs[1].sizes()==at::IntArrayRef({5120}) && inputs[2].sizes()==inputs[1].sizes(), "Gamma ABI");
  TORCH_CHECK(inputs[3].sizes()==at::IntArrayRef({1,585,1,128}) && inputs[4].sizes()==inputs[3].sizes(), "Coefficient ABI");
  return stream;
}
struct Call {
  std::array<Tensor,5> source;
  std::array<std::vector<Tensor>,4> ins, outs;
  std::vector<Tensor> owners;
  Tensor send, owner, qrope, krope, v;
};
std::shared_ptr<Call> allocate_call(const std::array<Tensor,5>& inputs, const Tensor* arena, bool with_send) {
  auto c=std::make_shared<Call>();c->source=inputs;c->owners.assign(inputs.begin(),inputs.end());
  auto alloc=[&](at::IntArrayRef shape,at::ScalarType dtype) {
    auto t=npu::empty_with_format(shape,dtype,{},inputs[0].device(),{},2);
    dense(t,dtype);c->owners.push_back(t);return t;
  };
  auto qn=alloc({1,585,5120},at::kBFloat16), kn=alloc({1,585,5120},at::kBFloat16);
  auto qs=alloc({1,585,1},at::kFloat), ks=alloc({1,585,1},at::kFloat);
  c->qrope=alloc({1,585,40,128},at::kBFloat16);c->krope=alloc({1,585,40,128},at::kBFloat16);
  auto parts=inputs[0].split(5120,-1);c->v=parts[2].view({1,585,40,128});
  c->ins={std::vector<Tensor>{parts[0],inputs[1]},std::vector<Tensor>{parts[1],inputs[2]},
          std::vector<Tensor>{qn.view({1,585,40,128}),inputs[3],inputs[4]},
          std::vector<Tensor>{kn.view({1,585,40,128}),inputs[3],inputs[4]}};
  c->outs={std::vector<Tensor>{qn,qs},std::vector<Tensor>{kn,ks},
           std::vector<Tensor>{c->qrope},std::vector<Tensor>{c->krope}};
  if(with_send) {
    c->owner=arena ? *arena : alloc({OWNER_ELEMENTS},at::kBFloat16);
    exact_arena(c->owner);
    if(arena)c->owners.push_back(*arena);
    c->send=c->owner.as_strided({8,585,3,1,5,128},{1123200,1920,640,640,128,1},256);
    c->owners.insert(c->owners.end(),{c->send,c->v});
  }
  return c;
}
struct Stage {
  aclOpExecutor* executor=nullptr;
  bool repeatable=false;
  int64_t stage1_status=-999,repeatable_status=-999;
  uint64_t bytes=0;
  Tensor workspace;
  std::vector<aclTensor*> ins,outs;
  std::vector<I> in_keys,out_keys;
};
enum class State { NEW, RESERVED, RECORDED, FAILED };
struct Plan {
  std::array<Stage,4> stages;
  aclrtEvent event=nullptr;
  State state=State::NEW;
  int64_t generation=0,recorded_generation=0;
  std::shared_ptr<Call> active;
};
struct Pool {
  std::mutex mutex;
  int64_t handle=0,device=-1,producer_tid=0,consumer_tid=0;
  aclrtContext ctx=nullptr;
  aclrtStream aclstream=nullptr;
  std::array<I,5> keys;
  std::vector<std::unique_ptr<Plan>> plans;
  std::shared_ptr<Call> cold_owners;
  std::vector<std::shared_ptr<Call>> quarantined;
  bool accepting=false,closed=false,poisoned=false,native_closed=false,finalizing=false;
  int64_t capacity=0,initialized=0,workspace_initial=0,workspace_resident=0;
  std::vector<std::shared_ptr<Call>> queued_calls;
  // Sized once by create. Consumer transfers only into empty slots; producer
  // detaches them after admission, then destroys the detached calls outside lock.
  std::vector<std::shared_ptr<Call>> retired;
  int64_t retirement_capacity=0,retired_head=0,retired_count=0,retired_high_water=0;
  int64_t retired_total=0,retirement_overflows=0,producer_reclaimed=0,reclaim_tid=0;
  int64_t finalize_released_calls=0,finalize_tid=0;
  int64_t submitted=0,callbacks_done=0,pending=0,pending_peak=0;
  int64_t queries=0,not_ready=0,reuses=0,used_peak=0,generations=0,exhaustions=0;
  int64_t last_error=0,last_step=-1;
};
// Deliberately no static destructor: failed pools/opaque executors must remain
// alive until worker termination, never destruct on an arbitrary Python thread.
struct Registry {std::mutex mutex;int64_t next=1;std::map<int64_t,std::shared_ptr<Pool>> pools;};
Registry& registry() {static auto* r=new Registry;return *r;}
std::shared_ptr<Pool> lookup(int64_t h) {
  auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);
  auto it=r.pools.find(h);TORCH_CHECK(it!=r.pools.end(), "Unknown pool handle");return it->second;
}
void check_status(int64_t code,const char* what,Pool& p,int64_t step=-1) {
  if(code) {p.last_error=code;p.last_step=step;TORCH_CHECK(false,what," failed: ",code," step ",step);}
}
void quarantine(const std::shared_ptr<Pool>& p,const std::shared_ptr<Call>& c={}) {
  std::lock_guard<std::mutex> lock(p->mutex);p->poisoned=true;p->accepting=false;
  if(!p->last_error)p->last_error=-1;
  for(auto& plan:p->plans)if(plan->state==State::RESERVED)plan->state=State::FAILED;
  if(c)p->quarantined.push_back(c);
}
aclTensor* make_descriptor(const Tensor& t) {
  const auto& d=descriptor(t);
  auto dtype=t.scalar_type()==at::kBFloat16 ? ACL_BF16 : ACL_FLOAT;
  auto* a=aclCreateTensor(t.sizes().data(),t.dim(),dtype,t.strides().data(),t.storage_offset(),
                         d.npu_format_,d.storage_sizes_.data(),d.storage_sizes_.size(),t.storage().data_ptr().get());
  TORCH_CHECK(a,"aclCreateTensor returned null");return a;
}
int64_t create(const Tensor& raw,const Tensor& qw,const Tensor& kw,const Tensor& cos,const Tensor& sin,int64_t capacity,int64_t retirement_capacity) {
  std::array<Tensor,5> inputs{raw,qw,kw,cos,sin};auto stream=admit(inputs);
  TORCH_CHECK(capacity>=1 && capacity<=64,"Pool capacity must be in [1,64]");
  TORCH_CHECK(retirement_capacity>=1 && retirement_capacity<=256,
              "Fixed retirement capacity must be in [1,256]");
  auto p=std::make_shared<Pool>();p->device=raw.device().index();p->producer_tid=tid();p->capacity=capacity;
  p->ctx=context();p->aclstream=stream.stream(false);
  p->retirement_capacity=retirement_capacity;p->retired.resize(retirement_capacity);
  for(size_t i=0;i<5;++i)p->keys[i]=signature(inputs[i]);
  {auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);p->handle=r.next++;r.pools.emplace(p->handle,p);}
  try {
    p->cold_owners=allocate_call(inputs,nullptr,false);
    for(int64_t n=0;n<capacity;++n) {
      p->plans.emplace_back(std::make_unique<Plan>());auto& plan=*p->plans.back();
      for(size_t j=0;j<4;++j) {
        auto& s=plan.stages[j];
        for(const auto& t:p->cold_owners->ins[j]) {s.ins.push_back(make_descriptor(t));s.in_keys.push_back(signature(t));}
        for(const auto& t:p->cold_owners->outs[j]) {s.outs.push_back(make_descriptor(t));s.out_keys.push_back(signature(t));}
        int rc=j<2 ? aclnnRmsNormGetWorkspaceSize(s.ins[0],s.ins[1],1e-6,s.outs[0],s.outs[1],&s.bytes,&s.executor)
                    : aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(s.ins[0],s.ins[1],s.ins[2],1,nullptr,s.outs[0],&s.bytes,&s.executor);
        s.stage1_status=rc;check_status(rc,"GetWorkspaceSize",*p,j);
        s.repeatable_status=aclSetAclOpExecutorRepeatable(s.executor);
        check_status(s.repeatable_status,"SetRepeatable",*p,j);s.repeatable=true;
        TORCH_CHECK(s.bytes<=uint64_t(INT64_MAX-p->workspace_initial),"Workspace total overflow");
        if(s.bytes) {
          s.workspace=npu::empty_with_format({int64_t(s.bytes)},at::kByte,{},raw.device(),{},2);
          dense(s.workspace,at::kByte);
          TORCH_CHECK(s.workspace.device()==raw.device() && s.workspace.is_contiguous() &&
                      s.workspace.storage_offset()==0 && s.workspace.storage().nbytes()==s.bytes,
                      "Workspace owner differs from requested extent");
        }
        p->workspace_initial+=s.bytes;p->workspace_resident+=s.bytes;
      }
      check_status(aclrtCreateEventExWithFlag(&plan.event,ACL_EVENT_CAPTURE_STREAM_PROGRESS),"CreateEvent",*p);
      ++p->initialized;
    }
    p->accepting=true;return p->handle;
  } catch(...) {quarantine(p);throw;}
}
void producer_check(const Pool& p,c10_npu::NPUStream stream) {
  TORCH_CHECK(tid()==p.producer_tid,"Foreign producer thread");
  TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx()==c10_npu::CaptureStatus::None,
              "No graph capture for producer or lifecycle entry");
  TORCH_CHECK(context()==p.ctx && stream.stream(false)==p.aclstream && stream.device_index()==p.device,"Pool context/stream/device changed");
}
void consumer_check(Pool& p) {
  TORCH_CHECK(!p.poisoned && !p.closed,"Pool poisoned/closed; retire workers");
  TORCH_CHECK(context()==p.ctx,"Consumer context changed");
  int32_t device=-1;check_status(aclrtGetDevice(&device),"GetDevice",p);
  TORCH_CHECK(device==p.device,"Consumer device changed");
  if(!p.consumer_tid)p.consumer_tid=tid();
  TORCH_CHECK(tid()==p.consumer_tid,"Consumer thread changed");
}
bool completed(Plan& plan,Pool& p) {
  TORCH_CHECK(plan.state==State::RECORDED && plan.generation>0 &&
              plan.recorded_generation==plan.generation,"Unrecorded generation cannot be queried");
  aclrtEventRecordedStatus status=ACL_EVENT_RECORDED_STATUS_NOT_READY;
  ++p.queries;check_status(aclrtQueryEventStatus(plan.event,&status),"QueryEvent",p);
  TORCH_CHECK(status==ACL_EVENT_RECORDED_STATUS_COMPLETE || status==ACL_EVENT_RECORDED_STATUS_NOT_READY,"Unknown event status");
  if(status==ACL_EVENT_RECORDED_STATUS_NOT_READY) {++p.not_ready;return false;}return true;
}
// Caller has completed the original input/physical/key validation. No event
// query here: every retired call was transferred after consumer-observed COMPLETE.
void reclaim_on_producer(const std::shared_ptr<Pool>& p) {
  TORCH_CHECK(tid()==p->producer_tid,"Foreign reclamation thread");
  std::vector<std::shared_ptr<Call>> local;
  local.reserve(p->retirement_capacity); // Host allocation, on producer only.
  {
    std::lock_guard<std::mutex> lock(p->mutex);
    TORCH_CHECK(p->accepting && !p->poisoned && !p->closed,"Cannot reclaim unhealthy pool");
    TORCH_CHECK(p->retired_count>=0 && p->retired_count<=p->retirement_capacity &&
                p->retired_head>=0 && p->retired_head<p->retirement_capacity,"Invalid retirement ring");
    for(int64_t i=0;i<p->retired_count;++i)
      TORCH_CHECK(p->retired[(p->retired_head+i)%p->retirement_capacity],"Missing retained completed call");
    while(p->retired_count) {
      auto& item=p->retired[p->retired_head];
      local.push_back(std::move(item));
      p->retired_head=(p->retired_head+1)%p->retirement_capacity;--p->retired_count;
    }
  }
  auto reclaimed=int64_t(local.size());
  // Last Tensor release may enqueue allocator events. Never hold Pool mutex
  // or execute this path from the framework consumer callback.
  local.clear();
  if(reclaimed) {
    std::lock_guard<std::mutex> lock(p->mutex);
    p->producer_reclaimed+=reclaimed;p->reclaim_tid=tid();
  }
}
void reclaim_retired(int64_t h) {
  auto p=lookup(h);
  try {
    TORCH_CHECK(c10::InferenceMode::is_enabled() && !at::GradMode::is_enabled(),"Inference only");
    TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx()==c10_npu::CaptureStatus::None,"No graph capture");
    auto stream=c10_npu::getCurrentNPUStream(p->device);
    TORCH_CHECK(stream==c10_npu::getDefaultNPUStream(p->device),"Default stream only");
    producer_check(*p,stream);
    reclaim_on_producer(p); // No event query, native execution or synchronization.
  } catch(...) {quarantine(p);throw;}
}
void retire_completed_on_consumer(Plan& plan,Pool& p) {
  // selected RECORDED was checked COMPLETE by this same callback. NEW has no
  // owner. This function never resets/overwrites a non-empty shared owner.
  if(!plan.active) {TORCH_CHECK(plan.state==State::NEW,"Used plan lost its active owner");return;}
  TORCH_CHECK(plan.state==State::RECORDED,"Only completed recorded owner may retire");
  if(p.retired_count==p.retirement_capacity) {
    ++p.retirement_overflows;p.last_error=-3;
    TORCH_CHECK(false,"RETIREMENT_EXHAUSTED: fixed completed-owner queue full; retire workers");
  }
  auto index=(p.retired_head+p.retired_count)%p.retirement_capacity;
  TORCH_CHECK(!p.retired[index],"Retirement slot would overwrite an owner");
  p.retired[index]=std::move(plan.active);
  ++p.retired_count;++p.retired_total;
  p.retired_high_water=std::max(p.retired_high_water,p.retired_count);
}
std::tuple<Tensor,Tensor> run_impl(int64_t h,const Tensor& raw,const Tensor& qw,const Tensor& kw,
                                 const Tensor& cos,const Tensor& sin,const Tensor* arena) {
  auto p=lookup(h);std::shared_ptr<Call> c;
  try {
    std::array<Tensor,5> inputs{raw,qw,kw,cos,sin};auto stream=admit(inputs);producer_check(*p,stream);
    {std::lock_guard<std::mutex> lock(p->mutex);TORCH_CHECK(p->accepting && !p->poisoned,"Pool not accepting");}
    for(size_t i=0;i<5;++i)TORCH_CHECK(signature(inputs[i])==p->keys[i],"Input descriptor differs from cold key");
    if(arena) {
      exact_arena(*arena);
      TORCH_CHECK(arena->device()==raw.device(),"Test arena device differs");
      for(const auto& t:inputs)at::assert_no_overlap(*arena,t);
      std::lock_guard<std::mutex> lock(p->mutex);
      // Testing may inject an arena, but may not alias any still-owned call.
      // No event query on this producer thread, even if an arena is likely done.
      for(const auto& queued:p->queued_calls)at::assert_no_overlap(*arena,queued->owner);
      for(const auto& plan:p->plans)if(plan->active)at::assert_no_overlap(*arena,plan->active->owner);
      for(const auto& retired:p->retired)if(retired)at::assert_no_overlap(*arena,retired->owner);
    }
    c=allocate_call(inputs,arena,true);
    // Do all descriptor work on producer, before callback or any native write.
    for(size_t j=0;j<4;++j) {
      const auto& s=p->plans[0]->stages[j];
      for(size_t i=0;i<c->ins[j].size();++i)TORCH_CHECK(signature(c->ins[j][i])==s.in_keys[i],"Native input key changed");
      for(size_t i=0;i<c->outs[j].size();++i)TORCH_CHECK(signature(c->outs[j][i])==s.out_keys[i],"Native output key changed");
    }
    // Private6 are retained by queued_calls/plan.active/retired until the
    // matching RECORDED generation is COMPLETE. Keep the full owner vector.
    // Verify the actual allocation order, including caller-visible aliases,
    // before omitting any allocator stream registration.
    TORCH_CHECK(c->owners.size()==14,"Private-retention owner count changed");
    std::array<Tensor,6> private_owners{c->outs[0][0],c->outs[1][0],
      c->outs[0][1],c->outs[1][1],c->qrope,c->krope};
    for(size_t i=0;i<5;++i)
      TORCH_CHECK(c->owners[i].unsafeGetTensorImpl()==inputs[i].unsafeGetTensorImpl(),
                  "Private-retention source owner order changed");
    for(size_t i=0;i<6;++i) {
      const auto& t=c->owners[5+i];
      TORCH_CHECK(t.unsafeGetTensorImpl()==private_owners[i].unsafeGetTensorImpl(),
                  "Private-retention temporary owner order changed");
      for(size_t j=0;j<14;++j)if(j!=5+i)
        TORCH_CHECK(t.storage().unsafeGetStorageImpl()!=c->owners[j].storage().unsafeGetStorageImpl(),
                    "Private temporary aliases another Call owner");
    }
    TORCH_CHECK(c->owners[11].unsafeGetTensorImpl()==c->owner.unsafeGetTensorImpl() &&
                c->owners[12].unsafeGetTensorImpl()==c->send.unsafeGetTensorImpl() &&
                c->owners[13].unsafeGetTensorImpl()==c->v.unsafeGetTensorImpl(),
                "Private-retention output/send/V owner order changed");
    reclaim_on_producer(p);
    for(size_t i=0;i<c->owners.size();++i) {
      if(i>=5 && i<11)continue;
      c10_npu::NPUCachingAllocator::recordStream(c->owners[i].storage().data_ptr(),stream);
    }
    {std::lock_guard<std::mutex> lock(p->mutex);p->queued_calls.push_back(c);++p->submitted;++p->pending;p->pending_peak=std::max(p->pending_peak,p->pending);}
    at_npu::native::OpCommand::RunOpApiV2("lingbot_qk_private_retention_v3",
      [p,c]()->int {
        try {
          std::lock_guard<std::mutex> lock(p->mutex);consumer_check(*p);Plan* selected=nullptr;
          for(auto& item:p->plans) {
            auto& plan=*item;
            if(plan.state==State::NEW) {selected=&plan;break;}
            if(plan.state==State::RECORDED && completed(plan,*p)) {selected=&plan;++p->reuses;break;}
          }
          if(!selected) {++p->exhaustions;p->last_error=-2;TORCH_CHECK(false,"POOL_EXHAUSTED: no completed plan; retire workers");}
          auto& plan=*selected;retire_completed_on_consumer(plan,*p);
          TORCH_CHECK(!plan.active,"Selected active owner not transferred");
          plan.active=c;plan.state=State::RESERVED;
          auto queued=std::find(p->queued_calls.begin(),p->queued_calls.end(),c);
          TORCH_CHECK(queued!=p->queued_calls.end(),"Missing queued owner");p->queued_calls.erase(queued);
          ++plan.generation;++p->generations;plan.recorded_generation=0;
          for(size_t j=0;j<4;++j) {
            auto& s=plan.stages[j];
            for(size_t i=0;i<s.ins.size();++i)
              check_status(aclSetInputTensorAddr(s.executor,i,s.ins[i],c->ins[j][i].storage().data_ptr().get()),"SetInputAddr",*p,j);
            for(size_t i=0;i<s.outs.size();++i)
              check_status(aclSetOutputTensorAddr(s.executor,i,s.outs[i],c->outs[j][i].storage().data_ptr().get()),"SetOutputAddr",*p,j);
          }
          for(size_t j=0;j<4;++j) {
            auto& s=plan.stages[j];void* workspace=s.bytes ? s.workspace.data_ptr() : nullptr;
            auto rc=j<2 ? aclnnRmsNorm(workspace,s.bytes,s.executor,p->aclstream)
                         : aclnnRotaryPositionEmbeddingV2(workspace,s.bytes,s.executor,p->aclstream);
            check_status(rc,"NativeStage2",*p,j);
          }
          check_status(lingbot_recovery_send_launch(p->aclstream,c->qrope.data_ptr(),c->krope.data_ptr(),
                                                   c->v.data_ptr(),c->send.data_ptr()),"ExactPack",*p);
          check_status(aclrtRecordEvent(plan.event,p->aclstream),"RecordEvent",*p);
          plan.recorded_generation=plan.generation;plan.state=State::RECORDED;
          int64_t used=0;for(const auto& item:p->plans)used+=item->state!=State::NEW;
          p->used_peak=std::max(p->used_peak,used);++p->callbacks_done;--p->pending;
          return 0;
        } catch(...) {quarantine(p,c);throw;}
      });
    return {c->send,c->owner};
  } catch(...) {quarantine(p,c);throw;}
}
std::tuple<Tensor,Tensor> run(int64_t h,const Tensor& raw,const Tensor& qw,const Tensor& kw,const Tensor& cos,const Tensor& sin) {
  return run_impl(h,raw,qw,kw,cos,sin,nullptr);
}
std::tuple<Tensor,Tensor> run_arena(int64_t h,const Tensor& raw,const Tensor& qw,const Tensor& kw,const Tensor& cos,const Tensor& sin,const Tensor& arena) {
  return run_impl(h,raw,qw,kw,cos,sin,&arena);
}
void close_pool(int64_t h) {
  auto p=lookup(h);
  try {
    producer_check(*p,c10_npu::getCurrentNPUStream(p->device));
    {std::lock_guard<std::mutex> lock(p->mutex);TORCH_CHECK(p->accepting && !p->closed && !p->poisoned && !p->native_closed && !p->finalizing,"Close requires healthy accepting pool");p->accepting=false;}
    // Caller must drain outside measurement first. This callback never waits.
    at_npu::native::OpCommand::RunOpApiV2("lingbot_qk_private_retention_close_v3",[p]()->int {
      try {
        std::lock_guard<std::mutex> lock(p->mutex);consumer_check(*p);
        TORCH_CHECK(p->pending==0 && p->queued_calls.empty(),"Pending callback at close");
        for(auto& item:p->plans)TORCH_CHECK(item->state==State::NEW || completed(*item,*p),"Incomplete plan at close; retire workers");
        for(auto& item:p->plans) {
          for(auto& s:item->stages) {
            TORCH_CHECK(s.repeatable && s.executor,"Uninitialized executor at close");
            check_status(aclDestroyAclOpExecutor(s.executor),"DestroyExecutor",*p);s.executor=nullptr;
            for(auto*& a:s.ins) {check_status(aclDestroyTensor(a),"DestroyInputDescriptor",*p);a=nullptr;}
            for(auto*& a:s.outs) {check_status(aclDestroyTensor(a),"DestroyOutputDescriptor",*p);a=nullptr;}
            // Keep framework workspace and all Call/template owners alive.
            // Final release is an explicit producer operation after a drain.
          }
          check_status(aclrtDestroyEvent(item->event),"DestroyEvent",*p);item->event=nullptr;
        }
        p->native_closed=true;return 0;
      } catch(...) {quarantine(p);throw;}
    });
  } catch(...) {quarantine(p);throw;}
}
// Caller drains after close, then invokes from the original producer thread.
// The consumer has destroyed native SDK resources but no framework Tensor owner.
void finalize_close(int64_t h) {
  auto p=lookup(h);
  std::vector<std::shared_ptr<Call>> calls;
  std::vector<Tensor> workspaces;
  try {
    producer_check(*p,c10_npu::getCurrentNPUStream(p->device));
    // Reserve before touching ownership; no framework allocation or queue task.
    calls.reserve(p->capacity+p->retirement_capacity+1);workspaces.reserve(p->capacity*4);
    int64_t retired_released=0;
    {
      std::lock_guard<std::mutex> lock(p->mutex);
      TORCH_CHECK(!p->accepting && p->native_closed && !p->closed && !p->finalizing &&
                  !p->poisoned && p->pending==0 && p->queued_calls.empty(),
                  "finalize_close requires drained successful SDK close; no retry");
      p->finalizing=true;
      // All native resources are already gone and every plan was COMPLETE.
      for(auto& item:p->plans) {
        TORCH_CHECK(item->event==nullptr,"Native event remains before finalize");
        for(auto& stage:item->stages) {
          TORCH_CHECK(stage.executor==nullptr,"Native executor remains before finalize");
          if(stage.workspace.defined())workspaces.push_back(std::move(stage.workspace));
        }
        if(item->active)calls.push_back(std::move(item->active));
      }
      while(p->retired_count) {
        auto& item=p->retired[p->retired_head];
        TORCH_CHECK(item,"Missing retired owner at finalize");
        calls.push_back(std::move(item));++retired_released;
        p->retired_head=(p->retired_head+1)%p->retirement_capacity;--p->retired_count;
      }
      if(p->cold_owners)calls.push_back(std::move(p->cold_owners));
    }
    auto released_calls=int64_t(calls.size());
    calls.clear();workspaces.clear(); // Original producer, outside Pool mutex.
    {
      std::lock_guard<std::mutex> lock(p->mutex);
      p->producer_reclaimed+=retired_released;
      if(retired_released)p->reclaim_tid=tid();
      p->finalize_released_calls=released_calls;p->finalize_tid=tid();
      p->workspace_resident=0;p->closed=true;p->finalizing=false;
    }
  } catch(...) {
    // Retain anything already detached before an invariant failure. This is
    // host-only quarantine, never consumer-side recovery or a release retry.
    {
      std::lock_guard<std::mutex> lock(p->mutex);
      for(auto& call:calls)if(call)p->quarantined.push_back(std::move(call));
      if(!workspaces.empty()) {
        auto keep=std::make_shared<Call>();keep->owners=std::move(workspaces);
        p->quarantined.push_back(std::move(keep));
      }
    }
    quarantine(p);throw;
  }
}
std::vector<std::string> stat_names() {
  return {"capacity","initialized","workspace_initial_bytes","workspace_resident_bytes","submitted",
    "callbacks_done","pending","pending_peak","event_queries","event_not_ready","plan_reuses",
    "used_plan_peak","generations","pool_exhaustions","producer_tid","consumer_tid","accepting",
    "closed","poisoned","last_error","last_step","retained_call_slots",
    "retirement_capacity","retired_count","retired_high_water","retired_total","retirement_overflows",
    "producer_reclaimed_calls","producer_reclaim_tid","native_closed","finalizing","finalize_released_calls","finalize_tid"};
}
I stats(int64_t h) {
  auto p=lookup(h);std::lock_guard<std::mutex> lock(p->mutex);
  int64_t retained=0;for(const auto& item:p->plans)retained+=bool(item->active);
  return {p->capacity,p->initialized,p->workspace_initial,p->workspace_resident,p->submitted,
    p->callbacks_done,p->pending,p->pending_peak,p->queries,p->not_ready,p->reuses,p->used_peak,p->generations,
    p->exhaustions,p->producer_tid,p->consumer_tid,p->accepting,p->closed,p->poisoned,p->last_error,p->last_step,retained,
    p->retirement_capacity,p->retired_count,p->retired_high_water,p->retired_total,p->retirement_overflows,
    p->producer_reclaimed,p->reclaim_tid,p->native_closed,p->finalizing,p->finalize_released_calls,p->finalize_tid};
}
I handles() {
  auto& r=registry();std::lock_guard<std::mutex> lock(r.mutex);I out;
  for(const auto& item:r.pools)out.push_back(item.first);return out;
}
I initialization_status(int64_t h) {
  auto p=lookup(h);std::lock_guard<std::mutex> lock(p->mutex);I out;
  for(size_t n=0;n<p->plans.size();++n)for(size_t j=0;j<4;++j) {
    const auto& s=p->plans[n]->stages[j];
    out.insert(out.end(),{int64_t(n),int64_t(j),s.stage1_status,s.repeatable_status,int64_t(s.bytes)});
  }
  return out;
}
std::vector<std::string> dependencies() {
  std::vector<std::string> paths;
  for(auto* f:{reinterpret_cast<void*>(&aclnnRmsNormGetWorkspaceSize),reinterpret_cast<void*>(&aclnnRmsNorm),
      reinterpret_cast<void*>(&aclnnRotaryPositionEmbeddingV2GetWorkspaceSize),reinterpret_cast<void*>(&aclnnRotaryPositionEmbeddingV2),
      reinterpret_cast<void*>(&aclSetAclOpExecutorRepeatable),reinterpret_cast<void*>(&aclDestroyAclOpExecutor),
      reinterpret_cast<void*>(&aclSetInputTensorAddr),reinterpret_cast<void*>(&aclSetOutputTensorAddr),
      reinterpret_cast<void*>(&aclCreateTensor),reinterpret_cast<void*>(&aclDestroyTensor),
      reinterpret_cast<void*>(&aclrtCreateEventExWithFlag),reinterpret_cast<void*>(&aclrtRecordEvent),
      reinterpret_cast<void*>(&aclrtQueryEventStatus),reinterpret_cast<void*>(&aclrtDestroyEvent),
      reinterpret_cast<void*>(&lingbot_recovery_send_launch)}) {
    Dl_info info{};TORCH_CHECK(dladdr(f,&info) && info.dli_fname,"Unbound SDK symbol");paths.emplace_back(info.dli_fname);
  }
  return paths;
}
} // namespace
TORCH_LIBRARY(lingbot_qk_private_retention_v3,m) {
  m.def("create(Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine, int capacity, int retirement_capacity) -> int",&create);
  m.def("run(int handle, Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine) -> (Tensor(a), Tensor(a))",&run);
  m.def("run_with_test_arena(int handle, Tensor raw, Tensor qw, Tensor kw, Tensor cosine, Tensor sine, Tensor(a!) arena) -> (Tensor(a!), Tensor(a!))",&run_arena);
  m.def("close(int handle) -> ()",&close_pool);
  m.def("finalize_close(int handle) -> ()",&finalize_close);
  m.def("reclaim_retired(int handle) -> ()",&reclaim_retired);
  m.def("stat_names() -> str[]",&stat_names);
  m.def("stats(int handle) -> int[]",&stats);
  m.def("handles() -> int[]",&handles);
  m.def("initialization_status(int handle) -> int[]",&initialization_status);
  m.def("dependencies() -> str[]",&dependencies);
}
