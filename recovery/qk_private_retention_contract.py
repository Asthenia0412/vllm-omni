"""CPU-only exact source delta for independent private-retention v3."""
from __future__ import annotations

import hashlib
from pathlib import Path

PARENT_SOURCE_SHA256 = 'fd65a7c9856f1e8726cd7350b257a0f9e11bde570632405cbb25fd179120e724'
NAMESPACE = 'lingbot_qk_private_retention_v3'
BUILD_SCHEMA = 'lingbot_qk_private_retention_cpu_build_v3'
REGISTRATION_POLICY = 'matched_generation_call_retention_private6_v3'
OWNER_ORDER = ('raw', 'qw', 'kw', 'cosine', 'sine', 'qn', 'kn', 'qs', 'ks',
               'qrope', 'krope', 'owner', 'send', 'v')
SKIPPED_OWNER_INDICES = tuple(range(5, 11))
RETAINED_OWNER_INDICES = (0, 1, 2, 3, 4, 11, 12, 13)
ORIGINAL_REGISTRATION = ('    reclaim_on_producer(p);\n'
    '    for(const auto& t:c->owners)c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);')
PRIVATE_REGISTRATION = '''    // Private6 are retained by queued_calls/plan.active/retired until the
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
    }'''


def need(value, message):
    if not value:
        raise ValueError('QK private-retention source contract: ' + message)


def derive_source(parent):
    """Only registration and independently named symbols may differ from v2."""
    need(hashlib.sha256(parent).hexdigest() == PARENT_SOURCE_SHA256,
         'frozen parent v2 source changed')
    text = parent.decode()
    need(text.count(ORIGINAL_REGISTRATION) == 1, 'registration anchor differs')
    text = text.replace(ORIGINAL_REGISTRATION, PRIVATE_REGISTRATION)
    text = text.replace('// Isolated v2 ownership candidate. No production registration or inherited qualification.',
                        '// Independent private-retention v3. No production registration or inherited qualification.')
    text = text.replace('lingbot_qk_repeatable_pool_close_v2', 'lingbot_qk_private_retention_close_v3')
    return text.replace('lingbot_qk_repeatable_pool_v2', NAMESPACE).encode()


def validate_source(parent_path, candidate_path):
    parent = Path(parent_path).read_bytes()
    candidate = Path(candidate_path).read_bytes()
    need(candidate == derive_source(parent), 'candidate differs outside the exact private6 delta')
    return dict(parent_source_sha256=PARENT_SOURCE_SHA256,
                candidate_source_sha256=hashlib.sha256(candidate).hexdigest(),
                registration_policy=REGISTRATION_POLICY, owner_order=list(OWNER_ORDER),
                omitted_owner_indices=list(SKIPPED_OWNER_INDICES),
                retained_owner_indices=list(RETAINED_OWNER_INDICES),
                private_registration_count=6, owner_count=14,
                owners_retained=True, arithmetic_unchanged=True,
                event_generation_retirement_quarantine_close_unchanged=True)
