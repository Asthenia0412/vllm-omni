// SPDX-License-Identifier: Apache-2.0
// Independent host-only per-layer metadata guard; no inherited qualification.
#include <ATen/ATen.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include <array>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {
using at::Tensor;
using I = std::vector<int64_t>;
namespace npu = at_npu::native::custom_ops;

void producer_check(const std::thread::id& creator) {
  TORCH_CHECK(std::this_thread::get_id() == creator, "StageB guard: foreign consumer thread");
  TORCH_CHECK(c10::InferenceMode::is_enabled() && !c10::GradMode::is_enabled(),
              "StageB guard: frozen inference/no_grad required");
}

bool unblocked(int64_t format) {
  return format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW || format == ACL_FORMAT_NCDHW;
}

const torch_npu::NPUStorageDesc& descriptor(const Tensor& t) {
  // CPU/undefined is rejected BEFORE reading or casting an NPU StorageImpl.
  TORCH_CHECK(t.defined() && t.device().type() == c10::DeviceType::PrivateUse1,
              "StageB guard requires a defined NPU tensor");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat,
              "StageB guard: BF16/FP32 only");
  const auto* s = static_cast<const torch_npu::NPUStorageImpl*>(t.storage().unsafeGetStorageImpl());
  const auto& d = s->npu_desc_;
  // Installed wheel does not export NPUStorageImpl RTTI. Pin this exact ABI,
  // retain the existing public-format cross-check, and never retag storage.
  TORCH_CHECK(int64_t(d.npu_format_) == npu::get_npu_format(t),
              "StageB guard: physical/public format mismatch");
  TORCH_CHECK(unblocked(d.npu_format_) && unblocked(d.origin_format_),
              "StageB guard: unsupported blocked physical format");
  return d;
}

void extent_check(const Tensor& t, const torch_npu::NPUStorageDesc& d) {
  TORCH_CHECK(!d.base_sizes_.empty() && d.base_sizes_.size() == d.base_strides_.size()
              && !d.storage_sizes_.empty(), "StageB guard: invalid physical descriptor dimensions");
  for (auto v : d.base_sizes_) TORCH_CHECK(v > 0, "StageB guard: invalid base extent");
  for (auto v : d.base_strides_) TORCH_CHECK(v >= 0, "StageB guard: invalid base stride");
  const uint64_t capacity = t.storage().nbytes() / t.element_size();
  uint64_t physical = 1;
  for (auto v : d.storage_sizes_) {
    TORCH_CHECK(v > 0 && uint64_t(v) <= std::numeric_limits<uint64_t>::max() / physical,
                "StageB guard: physical extent overflow");
    physical *= uint64_t(v);
  }
  TORCH_CHECK(physical <= capacity, "StageB guard: physical extent exceeds complete owner");
  int64_t high = t.storage_offset();
  TORCH_CHECK(high >= 0 && t.numel() > 0, "StageB guard: negative offset or empty view");
  for (int64_t i = 0; i < t.dim(); ++i) {
    TORCH_CHECK(t.size(i) > 0 && t.stride(i) >= 0, "StageB guard: invalid logical view");
    TORCH_CHECK(t.stride(i) == 0 || t.size(i)-1 <= INT64_MAX / t.stride(i),
                "StageB guard: view product overflow");
    const auto delta = (t.size(i)-1) * t.stride(i);
    TORCH_CHECK(high <= INT64_MAX-delta, "StageB guard: view extent overflow");
    high += delta;
  }
  TORCH_CHECK(uint64_t(high) < capacity && t.storage().data_ptr().get() != nullptr,
              "StageB guard: view exceeds complete allocation");
}

struct Version {
  bool enabled;
  uint32_t value;
  explicit Version(const Tensor& t) {
    const auto& counter = t.unsafeGetTensorImpl()->version_counter();
    enabled = counter.enabled();
    value = enabled ? counter.current_version() : 0;
    // value==0 in the disabled branch is not a version: enabled is always
    // compared separately. This preserves Python StageA's None semantics.
  }
  void check(const Tensor& t) const {
    const auto& counter = t.unsafeGetTensorImpl()->version_counter();
    TORCH_CHECK(counter.enabled() == enabled, "StageB guard: version availability changed");
    if (enabled) TORCH_CHECK(counter.current_version() == value, "StageB guard: version changed");
  }
};

struct Snapshot {
  // Both strong references are required: Tensor.set_ can change its StorageImpl.
  Tensor tensor;
  c10::Storage storage;
  c10::TensorImpl* tensor_impl;
  c10::StorageImpl* storage_impl;
  I shape, stride, base_sizes, base_strides, physical_sizes;
  int64_t offset, format, origin;
  size_t storage_bytes;
  const void* storage_ptr;
  const void* data_ptr;
  at::ScalarType dtype;
  c10::Device device;
  Version version;

  explicit Snapshot(const Tensor& value)
      : tensor(value), storage(value.storage()), tensor_impl(value.unsafeGetTensorImpl()),
        storage_impl(storage.unsafeGetStorageImpl()), shape(value.sizes().vec()),
        stride(value.strides().vec()), offset(value.storage_offset()),
        storage_bytes(storage.nbytes()), storage_ptr(storage.data_ptr().get()),
        data_ptr(value.const_data_ptr()), dtype(value.scalar_type()), device(value.device()), version(value) {
    const auto& d = descriptor(value);
    extent_check(value, d);
    format = d.npu_format_; origin = d.origin_format_;
    base_sizes.assign(d.base_sizes_.begin(), d.base_sizes_.end());
    base_strides.assign(d.base_strides_.begin(), d.base_strides_.end());
    physical_sizes.assign(d.storage_sizes_.begin(), d.storage_sizes_.end());
  }

  void check(const Tensor& actual) const {
    const auto& d = descriptor(actual);
    TORCH_CHECK(actual.unsafeGetTensorImpl() == tensor_impl
                && actual.storage().unsafeGetStorageImpl() == storage_impl,
                "StageB guard: TensorImpl/StorageImpl owner changed");
    TORCH_CHECK(actual.sizes() == at::IntArrayRef(shape) && actual.strides() == at::IntArrayRef(stride)
                && actual.storage_offset() == offset && actual.scalar_type() == dtype && actual.device() == device,
                "StageB guard: logical metadata changed");
    TORCH_CHECK(actual.storage().nbytes() == storage_bytes
                && actual.storage().data_ptr().get() == storage_ptr && actual.const_data_ptr() == data_ptr,
                "StageB guard: complete owner extent/pointers changed");
    TORCH_CHECK(int64_t(d.npu_format_) == format && int64_t(d.origin_format_) == origin
                && at::IntArrayRef(d.base_sizes_) == at::IntArrayRef(base_sizes)
                && at::IntArrayRef(d.base_strides_) == at::IntArrayRef(base_strides)
                && at::IntArrayRef(d.storage_sizes_) == at::IntArrayRef(physical_sizes),
                "StageB guard: physical descriptor changed");
    version.check(actual);
  }
  void check() const { check(tensor); }
};

// Precheck before Snapshot's initializer list can access any owner metadata.
const Tensor& admitted(const Tensor& t, at::ScalarType dtype, at::IntArrayRef shape) {
  descriptor(t);
  TORCH_CHECK(t.scalar_type() == dtype && t.sizes() == shape,
              "StageB guard: unsupported tensor geometry/dtype");
  return t;
}

class Frame : public torch::CustomClassHolder {
 public:
  explicit Frame(Tensor projection)
      : creator_(std::this_thread::get_id()), projection_(prepare(projection, creator_)) {}
  bool validate() const {
    producer_check(creator_);
    projection_.check();
    return true;
  }
  const c10::Device& device() const { return projection_.device; }
 private:
  static const Tensor& prepare(const Tensor& t, const std::thread::id& creator) {
    producer_check(creator);
    return admitted(t, at::kBFloat16, {1,1,6,5120});
  }
  const std::thread::id creator_;
  Snapshot projection_;
};

class Entry : public torch::CustomClassHolder {
 public:
  Entry(Tensor parameter, Tensor result, std::vector<Tensor> views)
      : creator_(std::this_thread::get_id()), parameter_(prepare_parameter(parameter, creator_)),
        result_(admitted(result, at::kFloat, {1,1,6,5120})) {
    TORCH_CHECK(views.size() == 6, "StageB guard: exactly six original views required");
    TORCH_CHECK(result_.device == parameter_.device, "StageB guard: mixed parameter/result devices");
    TORCH_CHECK(result_.storage_impl != parameter_.storage_impl, "StageB guard: result aliases parameter");
    views_.reserve(6);
    for (size_t i = 0; i < views.size(); ++i) {
      const auto& view = admitted(views[i], at::kFloat, {1,1,1,5120});
      TORCH_CHECK(view.device() == result_.device
                  && view.storage().unsafeGetStorageImpl() == result_.storage_impl
                  && view.strides() == at::IntArrayRef(result_.stride)
                  && view.storage_offset() == result_.offset + int64_t(i)*result_.stride[2],
                  "StageB guard: original view alias/order/layout differs");
      views_.emplace_back(view);
    }
  }

  bool validate(const Tensor& actual_parameter, const c10::intrusive_ptr<Frame>& frame) const {
    producer_check(creator_);
    TORCH_CHECK(frame, "StageB guard: actual typed Frame required");
    TORCH_CHECK(frame->device() == parameter_.device, "StageB guard: Frame/Entry device mismatch");
    // Keep original per-layer timing: all nine owners are checked immediately
    // before this layer consumes its cached tuple, not once at forward start.
    parameter_.check(actual_parameter);
    result_.check();
    for (const auto& view : views_) view.check();
    frame->validate();
    return true;
  }
 private:
  static const Tensor& prepare_parameter(const Tensor& t, const std::thread::id& creator) {
    producer_check(creator);
    return admitted(t, at::kBFloat16, {1,6,5120});
  }
  const std::thread::id creator_;
  Snapshot parameter_, result_;
  std::vector<Snapshot> views_;
};
} // namespace

TORCH_LIBRARY(lingbot_stageB_guard_v2, m) {
  m.class_<Frame>("Frame")
      .def(torch::init<Tensor>())
      .def("validate", &Frame::validate);
  m.class_<Entry>("Entry")
      .def(torch::init<Tensor, Tensor, std::vector<Tensor>>())
      .def("validate", &Entry::validate);
}
